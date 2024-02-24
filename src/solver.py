"""
This code is based on Facebook's HDemucs code: https://github.com/facebookresearch/demucs
"""

import json
import logging
from pathlib import Path
import os
import time
import wandb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio.transforms

from src.ddp import distrib
from src.data.datasets import PrHrSet, match_signal
from src.enhance import enhance, save_wavs, save_specs
from src.evaluate import evaluate, evaluate_on_saved_data
from src.model_serializer import SERIALIZE_KEY_BEST_STATES, SERIALIZE_KEY_MODELS, SERIALIZE_KEY_OPTIMIZERS,  \
    SERIALIZE_KEY_STATE, SERIALIZE_KEY_HISTORY, serialize
from src.models.discriminators import discriminator_loss, feature_loss, generator_loss
from src.models.stft_loss import MultiResolutionSTFTLoss
from src.utils import bold, copy_state, pull_metric, swap_state, LogProgress
from src.wandb_logger import create_wandb_table
from src.models.spec import spectro

logger = logging.getLogger(__name__)


GENERATOR_KEY = 'generator'

METRICS_KEY_EVALUATION_LOSS = 'evaluation_loss'
METRICS_KEY_BEST_LOSS = 'best_loss'

METRICS_KEY_LSD = 'Average lsd'
METRICS_KEY_VISQOL = 'Average visqol'


class Solver(object):
    def __init__(self, data, models, optimizers, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.tt_loader = data['tt_loader']
        self.args = args

        self.adversarial_mode = 'adversarial' in args.experiment and args.experiment.adversarial

        self.models = models
        self.dmodels = {k: distrib.wrap(model) for k, model in models.items()}
        self.model = self.models['generator']
        self.dmodel = self.dmodels['generator']


        self.optimizers = optimizers
        self.optimizer = optimizers['optimizer']
        if self.adversarial_mode:
            self.disc_optimizers = {'disc_optimizer': optimizers['disc_optimizer']}


        # Training config
        self.device = args.device
        self.epochs = args.epochs

        # Checkpoints
        self.continue_from = args.continue_from
        self.eval_every = args.eval_every
        self.cross_valid = args.cross_valid
        self.cross_valid_every = args.cross_valid_every
        self.checkpoint = args.checkpoint
        if self.checkpoint:
            self.checkpoint_file = Path(args.checkpoint_file)
            self.best_file = Path(args.best_file)
            logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.history_file = args.history_file

        self.best_states = None
        self.restart = args.restart
        self.history = []  # Keep track of loss
        self.samples_dir = args.samples_dir  # Where to save samples

        self.num_prints = args.num_prints  # Number of times to log per epoch
        self.dc_offset_loss_factor = args.dc_offset_loss_factor if 'dc_offset_loss_factor' in args else 25
        self.l1_loss_factor = args.l1_loss_factor if 'l1_loss_factor' in args else 100
        self.l2_loss_factor = args.l2_loss_factor if 'l2_loss_factor' in args else 100
        self.melgan_loss_factor = args.melgan_loss_factor if 'melgan_loss_factor' in args else 0.005
        self.msd_loss_factor = args.msd_loss_factor if 'msd_loss_factor' in args else 0.1
        self.mpd_loss_factor = args.mpd_loss_factor if 'mpd_loss_factor' in args else 0.1

        self.floatFormat = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

        if 'stft' in self.args.losses:
            self.mrstftloss = MultiResolutionSTFTLoss(factor_sc=args.stft_sc_factor,
                                                  factor_mag=args.stft_mag_factor).to(self.device)

        if 'discriminator_model' in self.args.experiment and \
                self.args.experiment.discriminator_model == 'hifi':
            self.melspec_transform = torchaudio.transforms.MelSpectrogram(
                                            self.args.experiment.hr_sr,
                                            **self.args.experiment.mel_spectrogram).to(self.device)

        self._reset()

    def _copy_models_states(self):
        states = {}
        for name, model in self.models.items():
            states[name] = copy_state(model.state_dict())
        return states

    def _load(self, package, load_best=False):
        if load_best:
            for name, model_package in package[SERIALIZE_KEY_BEST_STATES][SERIALIZE_KEY_MODELS].items():
                self.models[name].load_state_dict(model_package[SERIALIZE_KEY_STATE])
        else:
            for name, model_package in package[SERIALIZE_KEY_MODELS].items():
                self.models[name].load_state_dict(model_package[SERIALIZE_KEY_STATE])
            for name, opt_package in package[SERIALIZE_KEY_OPTIMIZERS].items():
                self.optimizers[name].load_state_dict(opt_package)


    def _reset(self):
        """_reset."""
        load_from = None
        load_best = False
        keep_history = True
        # Reset
        if self.checkpoint and self.checkpoint_file.exists() and not self.restart:
            load_from = self.checkpoint_file
        elif self.continue_from:
            load_from = self.continue_from
            load_best = self.args.continue_best
            keep_history = self.args.keep_history

        if load_from:
            logger.info(f'Loading checkpoint model: {load_from}')
            package = torch.load(load_from, 'cpu')
            self._load(package, load_best)
            if keep_history:
                self.history = package[SERIALIZE_KEY_HISTORY]
            self.best_states = package[SERIALIZE_KEY_BEST_STATES]


    def train(self):
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch + 1}: {info}")

        logger.info('-' * 70)
        logger.info("Trainable Params:")
        for name, model in self.models.items():
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mb = n_params * 4 / 2 ** 20
            logger.info(f"{name}: parameters: {n_params}, size: {mb} MB")

        best_loss = None
        if self.best_states == None: self.best_states = {}

        for epoch in range(len(self.history), self.epochs):
            # Train one epoch
            self.model.train()
            start = time.time()
            logger.info('-' * 70)
            logger.info("Training...")
            losses = self._run_one_epoch(epoch)
            logger_msg = f'Train Summary | End of Epoch {epoch + 1} | Time {time.time() - start:.2f}s | ' \
                         + ' | '.join([f'{k} Loss {v:.5f}' for k, v in losses.items()])
            logger.info(bold(logger_msg))
            losses = {k + '_loss': v for k, v in losses.items()}
            valid_losses = {}
            evaluation_loss = None

            evaluated_on_test_data = False

            if self.cross_valid and ((epoch + 1) % self.cross_valid_every == 0 or epoch == self.epochs - 1)\
                    and self.cv_loader:
                # Cross validation
                cross_valid_start = time.time()
                logger.info('-' * 70)
                logger.info('Cross validation...')
                self.model.eval()
                with torch.no_grad():
                    # if valid test equals all of test data, then
                    if self.args.valid_equals_test:
                        enhance_valid_data = (epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1 and self.tt_loader
                        valid_losses, enhanced_filenames = self._get_valid_losses_on_test_data(epoch,
                                                                                       enhance=enhance_valid_data)
                        evaluated_on_test_data = True
                    else:
                        valid_losses = self._run_one_epoch(epoch, cross_valid=True)
                self.model.train()
                evaluation_loss = valid_losses['evaluation']
                logger_msg = f'Validation Summary | End of Epoch {epoch + 1} | Time {time.time() - cross_valid_start:.2f}s | ' \
                             + ' | '.join([f'{k} Valid Loss {v:.5f}' for k, v in valid_losses.items()])
                logger.info(bold(logger_msg))
                valid_losses = {'valid_' + k + '_loss': v for k, v in valid_losses.items()}

                best_loss = min(pull_metric(self.history, 'valid_evaluation_loss') + [evaluation_loss])
                # Save the best model
                if evaluation_loss == best_loss:
                    logger.info(bold('New best valid loss %.4f'), evaluation_loss)
                    self.best_states = self._copy_models_states()
                    # a bit weird that we don't save/load optimizers' best states. Should we?
            elif not self.cross_valid and (len(self.best_states) == 0 or losses['total_loss'] < min(pull_metric(self.history, 'total_loss'))):
                    logger.info(bold('New best total loss %.4f'), losses['total_loss'])
                    self.best_states = self._copy_models_states()

            metrics = {**losses, **valid_losses}

            if evaluation_loss:
                metrics.update({METRICS_KEY_EVALUATION_LOSS: evaluation_loss})

            if best_loss:
                metrics.update({METRICS_KEY_BEST_LOSS: best_loss})

            # evaluate and enhance samples every 'eval_every' argument number of epochs
            # also evaluate on last epoch
            if ((epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1) and self.tt_loader:

                # Evaluate on the testset
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')
                # If best state exists and evalute_on_best configured, we switch to the best known model for testing.
                # Otherwise we use last state
                if self.args.evaluate_on_best and self.best_states:
                    logger.info('Loading best state.')
                    best_state = self.best_states[GENERATOR_KEY]
                else:
                    logger.info('Using last state.')
                    best_state = self.model.state_dict()
                with swap_state(self.model, best_state):
                    # enhance some samples
                    logger.info('Enhance and save samples...')
                    evaluation_start = time.time()

                    if evaluated_on_test_data:
                        logger.info('Samples already evaluated in cross validation, calculating metrics.')
                        enhanced_dataset = PrHrSet(self.args.samples_dir, enhanced_filenames)
                        enhanced_dataloader = distrib.loader(enhanced_dataset, batch_size=1, shuffle=False, num_workers=self.args.num_workers)
                        lsd, visqol = evaluate_on_saved_data(self.args, enhanced_dataloader, epoch)
                    elif self.args.joint_evaluate_and_enhance:
                        logger.info('Jointly evaluating and enhancing.')
                        lsd, visqol, enhanced_filenames = evaluate(self.args, self.tt_loader, epoch,
                                                              self.model)
                    else: # opposed to above cases, no spectrograms saved in samples directory.
                        enhanced_filenames = enhance(self.tt_loader, self.model, self.args)
                        enhanced_dataset = PrHrSet(self.args.samples_dir, enhanced_filenames)
                        enhanced_dataloader = DataLoader(enhanced_dataset, batch_size=1, shuffle=False)
                        lsd, visqol = evaluate_on_saved_data(self.args, enhanced_dataloader, epoch)

                    if epoch == self.epochs - 1 and self.args.log_results:
                        # log results at last epoch
                        if not 'enhanced_dataloader' in locals():
                            enhanced_dataset = PrHrSet(self.args.samples_dir, enhanced_filenames)
                            enhanced_dataloader = DataLoader(enhanced_dataset, batch_size=1, shuffle=False)

                        logger.info('logging results to wandb...')
                        create_wandb_table(self.args, enhanced_dataloader, epoch)


                    logger.info(bold(f'Evaluation Time {time.time() - evaluation_start:.2f}s'))

                metrics.update({METRICS_KEY_LSD: lsd, METRICS_KEY_VISQOL: visqol})



            wandb.log(metrics, step=epoch)
            self.history.append(metrics)
            info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
            logger.info('-' * 70)
            logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))

            if distrib.rank == 0:
                json.dump(self.history, open(self.history_file, "w"), indent=2)
                # Save model each epoch
                if self.checkpoint:
                    serialize(self.models, self.optimizers, self.history, self.best_states, self.args)
                    logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())


    def _run_one_epoch(self, epoch, cross_valid=False):
        total_losses = {}
        total_loss = 0
        losses = None
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # get a different order for distributed training, otherwise this will get ignored
        data_loader.epoch = epoch

        label = ["Train", "Valid"][cross_valid]
        name = label + f" | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader, updates=self.num_prints, name=name)

        # return_spec can be used to debug model and see explicit spectral output of model
        return_spec = 'return_spec' in self.args.experiment and self.args.experiment.return_spec

        enumeratedLogprog = enumerate(logprog)
        for i, data in enumeratedLogprog:
            total_generator_loss = 0
            with torch.autocast(device_type="cuda", dtype=self.floatFormat):

                lr, hr = [x.to(self.device) for x in data]

                if return_spec:
                    pr_time, pr_spec = self.dmodel(lr, return_spec=return_spec)
                    if cross_valid:
                        pr_time = match_signal(pr_time, hr.shape[-1])

                    hr_spec = self.dmodel._spec(hr, scale=True)

                    hr_reprs = {'time': hr, 'spec': hr_spec}
                    pr_reprs = {'time': pr_time, 'spec': pr_spec}
                else:
                    pr_time = self.dmodel(lr)
                    if cross_valid:
                        pr_time = match_signal(pr_time, hr.shape[-1])

                    hr_reprs = {'time': hr}
                    pr_reprs = {'time': pr_time}

                losses = self._get_losses(hr_reprs, pr_reprs)
                
                for loss_name, loss in losses['generator'].items():
                    total_generator_loss += loss

                total_loss += total_generator_loss.item()
                for loss_name, loss in losses['generator'].items():
                    total_loss_name = 'generator_' + loss_name
                    if total_loss_name in total_losses:
                        total_losses[total_loss_name] += loss.item()
                    else:
                        total_losses[total_loss_name] = loss.item()

                for loss_name, loss in losses['discriminator'].items():
                    total_loss_name = 'discriminator_' + loss_name
                    if total_loss_name in total_losses:
                        total_losses[total_loss_name] += loss.item()
                    else:
                        total_losses[total_loss_name] = loss.item()

                logprog.update(total_loss=format(total_loss / (i + 1), ".5f"))
                # Just in case, clear some memory
                if return_spec:
                    del pr_spec, hr_spec
                del pr_reprs, hr_reprs, pr_time, hr, lr

            # optimize model in training mode
            if not cross_valid:
                self._optimize(total_generator_loss)
                if self.adversarial_mode:
                    self._optimize_adversarial(losses['discriminator'])

        avg_losses = {'total': total_loss / (i + 1)}
        avg_losses.update({'evaluation': total_loss / (i + 1)})
        for loss_name, loss in total_losses.items():
            avg_losses.update({loss_name: loss / (i + 1)})

        return avg_losses

    # this function is very similar to _run_one_epoch, except it runs on *test* data-loader and returns the names of
    # enhanced files for later use. Kind of ugly...
    def _get_valid_losses_on_test_data(self, epoch, enhance):
        total_losses = {}
        total_loss = 0
        data_loader = self.tt_loader

        # get a different order for distributed training, otherwise this will get ignored
        data_loader.epoch = epoch

        name = f"Valid | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader, updates=self.num_prints, name=name)

        total_filenames = []

        enumeratedLogprog = enumerate(logprog)
        for i, data in enumeratedLogprog:
            (lr, lr_path), (hr, hr_path) = data
            lr = lr.to(self.device)
            hr = hr.to(self.device)

            filename = Path(hr_path[0]).stem
            total_filenames += filename
            if self.args.experiment.model == 'aero':
                hr_spec = self.model._spec(hr, scale=True).detach()
                pr_time, pr_spec, lr_spec = self.dmodel(lr, return_spec=True, return_lr_spec=True)
                pr_spec = pr_spec.detach()
                lr_spec = lr_spec.detach()
            else:
                nfft = self.args.experiment.nfft
                win_length = nfft // 4
                pr_time = self.model(lr)
                pr_spec = spectro(pr_time, n_fft=nfft, win_length=win_length)
                lr_spec = spectro(lr, n_fft=nfft, win_length=win_length)
                hr_spec = spectro(hr, n_fft=nfft, win_length=win_length)

            pr_time = match_signal(pr_time, hr.shape[-1])

            if enhance:
                save_wavs(pr_time, lr, hr, [os.path.join(self.args.samples_dir, filename)], self.args.experiment.lr_sr,
                          self.args.experiment.hr_sr)
                save_specs(lr_spec, pr_spec, hr_spec, os.path.join(self.args.samples_dir, filename))

            hr_reprs = {'time': hr, 'spec': hr_spec}
            pr_reprs = {'time': pr_time, 'spec': pr_spec}

            losses = self._get_losses(hr_reprs, pr_reprs)
            total_generator_loss = 0
            for loss_name, loss in losses['generator'].items():
                total_generator_loss += loss

            total_loss += total_generator_loss.item()
            for loss_name, loss in losses['generator'].items():
                total_loss_name = 'generator_' + loss_name
                if total_loss_name in total_losses:
                    total_losses[total_loss_name] += loss.item()
                else:
                    total_losses[total_loss_name] = loss.item()

            for loss_name, loss in losses['discriminator'].items():
                total_loss_name = 'discriminator_' + loss_name
                if total_loss_name in total_losses:
                    total_losses[total_loss_name] += loss.item()
                else:
                    total_losses[total_loss_name] = loss.item()

            logprog.update(total_loss=format(total_loss / (i + 1), ".5f"))
            # Just in case, clear some memory
            del pr_reprs, hr_reprs

        avg_losses = {'total': total_loss / (i + 1)}
        avg_losses.update({'evaluation': total_loss / (i + 1)})
        for loss_name, loss in total_losses.items():
            avg_losses.update({loss_name: loss / (i + 1)})

        return avg_losses, total_filenames if enhance else None


    def _get_losses(self, hr, pr):
        hr_time = hr['time']
        pr_time = pr['time']

        losses = {'generator': {}, 'discriminator': {}}
        with torch.autograd.set_detect_anomaly(True):
            if 'l1' in self.args.losses:
                losses['generator'].update({'l1': self.l1_loss_factor * F.l1_loss(pr_time, hr_time)})
            if 'l2' in self.args.losses:
                losses['generator'].update({'l2': self.l2_loss_factor * F.mse_loss(pr_time, hr_time)})
            if 'dc_offset'  in self.args.losses:
                pr_offset = torch.mean(pr_time)
                hr_offset = torch.mean(hr_time)
                dc_loss = self.dc_offset_loss_factor * abs((pr_offset - hr_offset) / 2) #max loss is -1 to 1
                losses['generator'].update({'dc_offset': dc_loss })
            if 'stft' in self.args.losses:
                stft_loss = self._get_stft_loss(pr_time, hr_time)
                losses['generator'].update({'stft': stft_loss})

            if self.adversarial_mode:
                if 'msd_melgan' in self.args.experiment.discriminator_models:
                    generator_losses, discriminator_loss = self._get_melgan_adversarial_loss(pr_time, hr_time)
                    if not self.args.experiment.only_features_loss:
                        losses['generator'].update({'adversarial_melgan': generator_losses['adversarial']})
                    if not self.args.experiment.only_adversarial_loss:
                        losses['generator'].update({'features_melgan': generator_losses['features']})
                    losses['discriminator'].update({'msd_melgan': discriminator_loss})
                if 'msd' in self.args.experiment.discriminator_models:
                    generator_losses, discriminator_loss = self._get_msd_adversarial_loss(pr_time, hr_time)
                    if not self.args.experiment.only_features_loss:
                        losses['generator'].update({'adversarial_msd': generator_losses['adversarial']})
                    if not self.args.experiment.only_adversarial_loss:
                        losses['generator'].update({'features_msd': generator_losses['features']})
                    losses['discriminator'].update({'msd': discriminator_loss})
                if 'mpd' in self.args.experiment.discriminator_models:
                    generator_losses, discriminator_loss = self._get_mpd_adversarial_loss(pr_time, hr_time)
                    if not self.args.experiment.only_features_loss:
                        losses['generator'].update({'adversarial_mpd': generator_losses['adversarial']})
                    if not self.args.experiment.only_adversarial_loss:
                        losses['generator'].update({'features_mpd': generator_losses['features']})
                    losses['discriminator'].update({'mpd': discriminator_loss})
                if 'hifi' in self.args.experiment.discriminator_models:
                    generator_loss, discriminator_loss = self._get_hifi_adversarial_loss(pr_time, hr_time)
                    losses['generator'].update({'adversarial_hifi': generator_loss})
                    losses['discriminator'].update({'hifi': discriminator_loss})
        return losses

    def _get_stft_loss(self, pr, hr):
        sc_loss, mag_loss = self.mrstftloss(
            pr.reshape([pr.shape[0], pr.shape[1] * pr.shape[2]]), 
            hr.reshape([hr.shape[0], hr.shape[1] * hr.shape[2]])
            )
        stft_loss = sc_loss + mag_loss
        return stft_loss

    def _get_melgan_adversarial_loss(self, pr, hr):

        discriminator = self.dmodels['msd_melgan']

        discriminator_fake_detached = discriminator(pr.detach())
        discriminator_real = discriminator(hr)
        discriminator_fake = discriminator(pr)

        total_loss_discriminator = self._get_melgan_discriminator_loss(discriminator_fake_detached, discriminator_real)
        generator_losses = self._get_melgan_generator_loss(discriminator_fake, discriminator_real)


        return generator_losses, total_loss_discriminator


    def _get_melgan_discriminator_loss(self, discriminator_fake, discriminator_real):
        discriminator_loss = 0
        for scale in discriminator_fake:
            discriminator_loss += self.melgan_loss_factor * F.relu(1 + scale[-1]).mean()

        for scale in discriminator_real:
            discriminator_loss += self.melgan_loss_factor * F.relu(1 - scale[-1]).mean()
        return discriminator_loss

    def _get_melgan_generator_loss(self, discriminator_fake, discriminator_real):
        features_loss = 0
        features_weights = 4.0 / (self.args.experiment.melgan_discriminator.n_layers + 1)
        discriminator_weights = 1.0 / self.args.experiment.melgan_discriminator.num_D
        weights = discriminator_weights * features_weights

        for i in range(self.args.experiment.melgan_discriminator.num_D):
            for j in range(len(discriminator_fake[i]) - 1):
                features_loss += self.melgan_loss_factor * weights * F.l1_loss(discriminator_fake[i][j], discriminator_real[i][j].detach())

        adversarial_loss = 0
        for scale in discriminator_fake:
            adversarial_loss += self.melgan_loss_factor * F.relu(1 - scale[-1]).mean()

        if 'only_adversarial_loss' in self.args.experiment and self.args.experiment.only_adversarial_loss:
            return {'adversarial': adversarial_loss}

        if 'only_features_loss' in self.args.experiment and self.args.experiment.only_features_loss:
            return {'features': self.args.experiment.features_loss_lambda * features_loss}

        return {'adversarial': adversarial_loss ,
                'features': self.args.experiment.features_loss_lambda * features_loss}


    def _get_hifi_adversarial_loss(self, pr, hr):
        mpd = self.dmodels['mpd']
        msd = self.dmodels['msd_hifi']

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = mpd(hr, pr.detach())
        loss_disc_f = discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = msd(hr, pr.detach())
        loss_disc_s = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        total_loss_discriminator = loss_disc_s + loss_disc_f

        # L1 Mel-Spectrogram Loss
        pr_mel = self.melspec_transform(pr)
        hr_mel = self.melspec_transform(hr)
        loss_mel = F.l1_loss(hr_mel, pr_mel) * self.args.experiment.mel_spec_loss_lambda

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(hr, pr)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(hr, pr)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s = generator_loss(y_ds_hat_g)

        if 'only_features_loss' in self.args.experiment and self.args.experiment.only_features_loss:
            total_loss_generator = loss_fm_s + loss_fm_f
        else:
            total_loss_generator = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        return total_loss_generator, total_loss_discriminator


    def _get_msd_adversarial_loss(self, pr, hr):
        msd = self.dmodels['msd']

        # discriminator loss
        y_ds_hat_r, y_ds_hat_g, _, _ = msd(hr, pr.detach())
        d_loss = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        # generator loss
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(hr, pr)
        g_feat_loss = feature_loss(fmap_s_r, fmap_s_g)
        g_adv_loss = generator_loss(y_ds_hat_g)


        if 'only_adversarial_loss' in self.args.experiment and self.args.experiment.only_adversarial_loss:
            return {'adversarial': self.msd_loss_factor * g_adv_loss}, d_loss

        if 'only_features_loss' in self.args.experiment and self.args.experiment.only_features_loss:
            return {'features': self.msd_loss_factor * self.args.experiment.features_loss_lambda * g_feat_loss}, d_loss

        return {'adversarial': self.msd_loss_factor * g_adv_loss,
                'features': self.msd_loss_factor * self.args.experiment.features_loss_lambda * g_feat_loss}, d_loss


    def _get_mpd_adversarial_loss(self, pr, hr):
        mpd = self.dmodels['mpd']

        # discriminator loss
        y_df_hat_r, y_df_hat_g, _, _ = mpd(hr, pr.detach())
        d_loss = discriminator_loss(y_df_hat_r, y_df_hat_g)

        # generator loss
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(hr, pr)
        g_feat_loss = feature_loss(fmap_f_r, fmap_f_g)
        g_adv_loss = generator_loss(y_df_hat_g)

        if 'only_adversarial_loss' in self.args.experiment and self.args.experiment.only_adversarial_loss:
            return {'adversarial': self.mpd_loss_factor * g_adv_loss}, d_loss

        if 'only_features_loss' in self.args.experiment and self.args.experiment.only_features_loss:
            return {'features': self.mpd_loss_factor * self.args.experiment.features_loss_lambda * g_feat_loss}, d_loss

        return {'adversarial': self.mpd_loss_factor * g_adv_loss,
                'features': self.mpd_loss_factor * self.args.experiment.features_loss_lambda * g_feat_loss}, d_loss


    def _optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _optimize_adversarial(self, discriminator_losses):
        total_disc_loss = sum(list(discriminator_losses.values()))
        disc_optimizer = self.disc_optimizers['disc_optimizer']
        disc_optimizer.zero_grad()
        total_disc_loss.backward()
        disc_optimizer.step()