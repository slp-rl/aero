import argparse
import os
import logging

import PIL
import torch
import wandb

from torchaudio.transforms import Spectrogram
from torch.utils.data import DataLoader

from src.ddp import distrib
from src.data.datasets import PrHrSet, match_signal
from src.enhance import save_wavs, save_specs
from src.log_results import log_results
from src.metrics import run_metrics
from src.models.spec import spectro
from src.utils import LogProgress, bold, convert_spectrogram_to_heatmap

logger = logging.getLogger(__name__)

WANDB_PROJECT_NAME = 'Bandwidth Extension'
WANDB_ENTITY = 'huji-dl-audio-lab'



def evaluate_lr_hr_pr_data(data, wandb_n_files_to_log, files_to_log, epoch, args):
    lr, hr, pr, filename = data
    filename = filename[0]
    hr_sr = args.experiment.hr_sr if 'experiment' in args else args.hr_sr
    if args.experiment.upsample:
        lr_sr = hr_sr
    else:
        lr_sr = args.experiment.lr_sr if 'experiment' in args else args.lr_sr
    # logger.info(f'evaluating on {filename}')
    if wandb_n_files_to_log == -1 or len(files_to_log) < wandb_n_files_to_log:
        files_to_log.append(filename)

    if args.device != 'cpu':
        hr = hr.cpu()
        pr = pr.cpu()

    hr_spec_path = os.path.join(args.samples_dir, filename + '_hr_spec.png')
    pr_spec_path = os.path.join(args.samples_dir, filename + '_pr_spec.png')
    lr_spec_path = os.path.join(args.samples_dir, filename + '_lr_spec.png')

    hr_spec = PIL.Image.open(hr_spec_path) if os.path.exists(hr_spec_path) else None
    pr_spec = PIL.Image.open(pr_spec_path) if os.path.exists(pr_spec_path) else None
    lr_spec = PIL.Image.open(lr_spec_path) if os.path.exists(lr_spec_path) else None

    pesq_i, stoi_i, snr_i, lsd_i, sisnr_i, visqol_i, estimate_i = run_metrics(hr, pr, args, filename)
    if filename in files_to_log:

        log_to_wandb(pr, hr, lr, pesq_i, stoi_i, snr_i, lsd_i, sisnr_i, visqol_i,
                     filename, epoch, lr_sr, hr_sr, lr_spec, pr_spec, hr_spec)

    return {'pesq': pesq_i,'stoi': stoi_i, 'snr': snr_i, 'lsd': lsd_i,
            'sisnr':sisnr_i, 'visqol': visqol_i, 'filename': filename}

from pathlib import Path

"""
This is for saving intermediate spectrogram output as well as final time signal output of model.
"""
def evaluate_lr_hr_data(data, model, wandb_n_files_to_log, files_to_log, epoch, args, enhance=True):
    (lr, lr_path), (hr, hr_path) = data
    lr, hr = lr.to(args.device), hr.to(args.device)
    hr_sr = args.experiment.hr_sr if 'experiment' in args else args.hr_sr
    if args.experiment.upsample:
        lr_sr = hr_sr
    else:
        lr_sr = args.experiment.lr_sr if 'experiment' in args else args.lr_sr
    with torch.no_grad():
        pr, pr_spec, lr_spec = model(lr, return_spec=True, return_lr_spec=True)
    pr = match_signal(pr, hr.shape[-1])
    hr_spec = model._spec(hr, scale=True)
    filename = Path(hr_path[0]).stem

    # logger.info(f'evaluating on {filename}')
    if wandb_n_files_to_log == -1 or len(files_to_log) < wandb_n_files_to_log:
        files_to_log.append(filename)

    if args.device != 'cpu':
        hr = hr.cpu()
        pr = pr.cpu()
        lr = lr.cpu()

    pesq_i, stoi_i, snr_i, lsd_i, sisnr_i, visqol_i, estimate_i = run_metrics(hr, pr, args, filename)
    if filename in files_to_log:
        log_to_wandb(estimate_i, hr, lr, pesq_i, stoi_i, snr_i, lsd_i, sisnr_i, visqol_i,
                     filename, epoch, lr_sr, hr_sr, lr_spec.cpu(), pr_spec.cpu(), hr_spec.cpu())

    if enhance:
        os.makedirs(args.samples_dir, exist_ok=True)
        lr_sr = args.experiment.hr_sr if args.experiment.upsample else args.experiment.lr_sr
        save_wavs(pr, lr, hr, [os.path.join(args.samples_dir, filename)], lr_sr, args.experiment.hr_sr)
        save_specs(lr_spec, pr_spec, hr_spec, os.path.join(args.samples_dir, filename))

    return {'pesq': pesq_i, 'stoi': stoi_i, 'snr': snr_i, 'lsd': lsd_i,
            'sisnr': sisnr_i, 'visqol': visqol_i, 'filename': filename}


def evaluate_on_saved_data(args, data_loader, epoch):
    total_pesq = 0
    total_stoi = 0
    total_lsd = 0
    total_sisnr = 0
    total_visqol = 0

    pesq_count = 0
    stoi_count = 0
    lsd_count = 0
    sisnr_count = 0
    visqol_count = 0

    total_cnt = 0

    files_to_log = []
    wandb_n_files_to_log = args.wandb.n_files_to_log if 'wandb' in args else args.wandb_n_files_to_log
    hr_sr = args.experiment.hr_sr if 'experiment' in args else args.hr_sr

    with torch.no_grad():
        iterator = LogProgress(logger, data_loader, name="Eval estimates")
        for i, data in enumerate(iterator):
            metrics_i = evaluate_lr_hr_pr_data(data, wandb_n_files_to_log, files_to_log, epoch, args)

            total_pesq += metrics_i['pesq']
            total_stoi += metrics_i['stoi']
            total_lsd += metrics_i['lsd']
            total_sisnr += metrics_i['sisnr']
            total_visqol += metrics_i['visqol']

            pesq_count += 1 if metrics_i['pesq'] != 0 else 0
            stoi_count += 1 if metrics_i['stoi'] != 0 else 0
            lsd_count += 1 if metrics_i['lsd'] != 0 else 0
            sisnr_count += 1 if metrics_i['sisnr'] != 0 else 0
            visqol_count += 1 if metrics_i['visqol'] != 0 else 0

            total_cnt += 1

    if pesq_count != 0:
        avg_pesq, = distrib.average([total_pesq / pesq_count], total_pesq)
    else:
        avg_pesq = 0
    if stoi_count != 0:
        avg_stoi, = distrib.average([total_stoi / stoi_count], stoi_count)
    else:
        avg_stoi = 0
    if lsd_count != 0:
        # avg_lsd, = distrib.average([total_lsd / lsd_count], lsd_count)
        avg_lsd, = [total_lsd / lsd_count]
    else:
        avg_lsd = 0
    if sisnr_count != 0:
        avg_sisnr, = distrib.average([total_sisnr / sisnr_count], sisnr_count)
    else:
        avg_sisnr = 0
    if visqol_count != 0:
        # avg_visqol, = distrib.average([total_visqol / visqol_count], visqol_count)
        avg_visqol, = [total_visqol / visqol_count]
    else:
        avg_visqol = 0

    logger.info(bold(
        f'{args.experiment.name}, {args.experiment.lr_sr}->{args.experiment.hr_sr}. Test set performance:PESQ={avg_pesq} ({pesq_count}/{total_cnt}), STOI={avg_stoi} ({stoi_count}/{total_cnt}),'
        f'LSD={avg_lsd} ({lsd_count}/{total_cnt}), SISNR={avg_sisnr} ({sisnr_count}/{total_cnt}),VISQOL={avg_visqol} ({visqol_count}/{total_cnt}).'))

    return avg_pesq, avg_stoi, avg_lsd, avg_sisnr, avg_visqol


def evaluate(args, data_loader, epoch, model):
    total_pesq = 0
    total_stoi = 0
    total_lsd = 0
    total_sisnr = 0
    total_visqol = 0

    pesq_count = 0
    stoi_count = 0
    lsd_count = 0
    sisnr_count = 0
    visqol_count = 0

    total_cnt = 0

    total_filenames = []

    files_to_log = []
    wandb_n_files_to_log = args.wandb.n_files_to_log if 'wandb' in args else args.wandb_n_files_to_log

    with torch.no_grad():
        iterator = LogProgress(logger, data_loader, name="Eval estimates")
        for i, data in enumerate(iterator):

            metrics_i = evaluate_lr_hr_data(data, model, wandb_n_files_to_log, files_to_log, epoch, args)
            total_pesq += metrics_i['pesq']
            total_stoi += metrics_i['stoi']
            total_lsd += metrics_i['lsd']
            total_sisnr += metrics_i['sisnr']
            total_visqol += metrics_i['visqol']

            total_filenames += metrics_i['filename']

            pesq_count += 1 if metrics_i['pesq'] != 0 else 0
            stoi_count += 1 if metrics_i['stoi'] != 0 else 0
            lsd_count += 1 if metrics_i['lsd'] != 0 else 0
            sisnr_count += 1 if metrics_i['sisnr'] != 0 else 0
            visqol_count += 1 if metrics_i['visqol'] != 0 else 0

            total_cnt += 1

    if pesq_count != 0:
        avg_pesq, = distrib.average([total_pesq / pesq_count], total_pesq)
    else:
        avg_pesq = 0
    if stoi_count != 0:
        avg_stoi, = distrib.average([total_stoi / stoi_count], stoi_count)
    else:
        avg_stoi = 0
    if lsd_count != 0:
        avg_lsd, = distrib.average([total_lsd / lsd_count], lsd_count)
    else:
        avg_lsd = 0
    if sisnr_count != 0:
        avg_sisnr, = distrib.average([total_sisnr / sisnr_count], sisnr_count)
    else:
        avg_sisnr = 0
    if visqol_count != 0:
        avg_visqol, = distrib.average([total_visqol / visqol_count], visqol_count)
    else:
        avg_visqol = 0


    logger.info(bold(
        f'{args.experiment.name}, {args.experiment.lr_sr}->{args.experiment.hr_sr}. Test set performance:PESQ={avg_pesq} ({pesq_count}/{total_cnt}), STOI={avg_stoi} ({stoi_count}/{total_cnt}),'
                                f'LSD={avg_lsd} ({lsd_count}/{total_cnt}), SISNR={avg_sisnr} ({sisnr_count}/{total_cnt}),VISQOL={avg_visqol} ({visqol_count}/{total_cnt}).'))
    return avg_pesq, avg_stoi, avg_lsd, avg_sisnr, avg_visqol, total_filenames


def log_to_wandb(pr_signal, hr_signal, lr_signal, pesq, stoi, snr, lsd, sisnr, visqol, filename, epoch, lr_sr, hr_sr, lr_spec=None, pr_spec=None, hr_spec=None):
    spectrogram_transform = Spectrogram()
    enhanced_spectrogram = spectrogram_transform(pr_signal).log2()[0, :, :].numpy()
    enhanced_spectrogram_wandb_image = wandb.Image(convert_spectrogram_to_heatmap(enhanced_spectrogram),
                                                   caption='PR')
    enhanced_wandb_audio = wandb.Audio(pr_signal.squeeze().numpy(), sample_rate=hr_sr, caption='PR')

    wandb_dict = {f'test samples/{filename}/pesq': pesq,
                  f'test samples/{filename}/stoi': stoi,
                  f'test samples/{filename}/snr': snr,
                  f'test samples/{filename}/lsd': lsd,
                  f'test samples/{filename}/sisnr': sisnr,
                  f'test samples/{filename}/visqol': visqol,
                  f'test samples/{filename}/spectrogram': enhanced_spectrogram_wandb_image,
                  f'test samples/{filename}/audio': enhanced_wandb_audio}

    if pr_spec is not None and hr_spec is not None and lr_spec is not None:
        if not isinstance(pr_spec, PIL.Image.Image):
            pr_spec = pr_spec.abs().pow(2).log2()[0,:,:].numpy()
            pr_spec = convert_spectrogram_to_heatmap(pr_spec)
        enhanced_pr_spectrogram_wandb_image = wandb.Image(pr_spec, caption='PR spec')
        wandb_dict.update({f'test samples/{filename}/pr_spec': enhanced_pr_spectrogram_wandb_image})

        if epoch <= 10:
            if not isinstance(hr_spec, PIL.Image.Image):
                hr_spec = hr_spec.abs().pow(2).log2()[0, :, :].numpy()
                hr_spec = convert_spectrogram_to_heatmap(hr_spec)
            enhanced_hr_spectrogram_wandb_image = wandb.Image(hr_spec, caption='HR spec')
            wandb_dict.update({f'test samples/{filename}/hr_spec': enhanced_hr_spectrogram_wandb_image})

            if not isinstance(lr_spec, PIL.Image.Image):
                lr_spec = lr_spec.abs().pow(2).log2()[0, :, :].numpy()
                lr_spec = convert_spectrogram_to_heatmap(lr_spec)
            enhanced_lr_spectrogram_wandb_image = wandb.Image(lr_spec, caption='LR spec')
            wandb_dict.update({f'test samples/{filename}/lr_spec': enhanced_lr_spectrogram_wandb_image})

    if epoch <= 10:
        hr_name = f'{filename}_hr'
        hr_enhanced_spectrogram = spectrogram_transform(hr_signal).log2()[0, :, :].numpy()
        hr_enhanced_spectrogram_wandb_image = wandb.Image(convert_spectrogram_to_heatmap(hr_enhanced_spectrogram),
                                                       caption='HR')
        hr_enhanced_wandb_audio = wandb.Audio(hr_signal.squeeze().numpy(), sample_rate=hr_sr, caption='HR')
        wandb_dict.update({f'test samples/{filename}/{hr_name}_spectrogram': hr_enhanced_spectrogram_wandb_image,
                               f'test samples/{filename}/{hr_name}_audio': hr_enhanced_wandb_audio})

        lr_name = f'{filename}_lr'
        lr_enhanced_spectrogram = spectrogram_transform(lr_signal).log2()[0, :, :].numpy()
        lr_enhanced_spectrogram_wandb_image = wandb.Image(convert_spectrogram_to_heatmap(lr_enhanced_spectrogram),
                                                          caption='LR')
        lr_enhanced_wandb_audio = wandb.Audio(lr_signal.squeeze().numpy(), sample_rate=lr_sr, caption='LR')
        wandb_dict.update({f'test samples/{filename}/{lr_name}_spectrogram': lr_enhanced_spectrogram_wandb_image,
                           f'test samples/{filename}/{lr_name}_audio': lr_enhanced_wandb_audio})

    wandb.log(wandb_dict,
              step=epoch)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('samples_dir', type=str)
    parser.add_argument('--device', nargs="?", default='cpu', type=str, choices=['cpu', 'cuda'])
    parser.add_argument('--lr_sr', nargs="?", default=8000, type=int)
    parser.add_argument('--hr_sr', nargs="?", default=16000, type=int)
    parser.add_argument('--num_workers', nargs="?", default=1, type=int)
    parser.add_argument('--wandb_mode', nargs="?", default='online', type=str)
    parser.add_argument('--wandb_n_files_to_log', nargs="?", default=10, type=int)
    parser.add_argument('--n_bins', nargs="?", default=5, type=int)
    parser.add_argument('--log_results', action='store_false')


    return parser

def update_args(args):
    d = vars(args)
    experiment = argparse.Namespace()
    experiment.name = 'nuwave-singlespeaker'
    experiment.lr_sr = args.lr_sr
    experiment.hr_sr = args.hr_sr
    d['experiment'] = experiment


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    samples_dir = args.samples_dir
    print(args)
    update_args(args)
    print(args)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)


    wandb_mode = os.environ['WANDB_MODE'] if 'WANDB_MODE' in os.environ.keys() else args.wandb_mode
    wandb.init(mode=wandb_mode, project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, config=args, group='single-speaker-8-16')

    data_set = PrHrSet(samples_dir)
    dataloader = DataLoader(data_set, batch_size=1, shuffle=False)
    avg_pesq, avg_stoi, avg_lsd, avg_sisnr, avg_visqol = evaluate_on_saved_data(args, dataloader, epoch=0)

    log_results(args, dataloader, epoch=0)

    print(f'pesq: {avg_pesq}, stoi: {avg_stoi}, lsd: {avg_lsd}, sisnr: {avg_sisnr}, visqol: {avg_visqol}')
    print('done evaluating.')