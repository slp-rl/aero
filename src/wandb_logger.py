import os

import PIL
import wandb
import logging

from torchaudio.functional import resample
from torchaudio.transforms import Spectrogram

from src.metrics import run_metrics
from src.utils import convert_spectrogram_to_heatmap

logger = logging.getLogger(__name__)

SPECTOGRAM_EPSILON = 1e-13


def _get_wandb_config(args):
    included_keys = ['eval_every', 'optim', 'lr', 'losses', 'epochs']
    wandb_config = {k: args[k] for k in included_keys}
    wandb_config.update(**args.experiment)
    wandb_config.update({'train': args.dset.train, 'test': args.dset.test})
    return wandb_config


def _init_wandb_run(args, train=True):
    tags = args.wandb.tags
    wandb_mode = os.environ['WANDB_MODE'] if 'WANDB_MODE' in os.environ.keys() else args.wandb.mode
    logger.info(f'current path: {os.getcwd()}, rank: {args.rank}')
    if args.ddp:
        experiment_name = args.experiment.name + f'-rank={args.rank}'
    else:
        experiment_name = args.experiment.name
    if train and args.ddp and args.wandb.resume:
        group_id_path = os.path.join(os.getcwd(), 'group_id.dat')
        if not os.path.exists(group_id_path):
            group_id = wandb.util.generate_id()
            with open(group_id_path, 'w+') as f:
                f.write(group_id)
        else:
            group_id = open(group_id_path).read()
        wandb.init(mode=wandb_mode, project=args.wandb.project_name, entity=args.wandb.entity,
                   config=_get_wandb_config(args),
                   group=os.path.basename(args.dset.name),
                   id=f"{group_id}-worker-{args.rank}", job_type="worker",
                   resume='allow', name=experiment_name,
                   tags=tags)
    else:
        wandb.init(mode=wandb_mode, project=args.wandb.project_name, entity=args.wandb.entity,
                   config=_get_wandb_config(args),
                   group=os.path.basename(args.dset.name), resume=args.wandb.resume, name=experiment_name,
                   tags=tags)


def log_data_to_wandb(pr_signal, hr_signal, lr_signal, lsd, visqol, filename, epoch, lr_sr, hr_sr, lr_spec=None, pr_spec=None, hr_spec=None):
    spectrogram_transform = Spectrogram()
    enhanced_spectrogram = spectrogram_transform(pr_signal).log2()[0, :, :].numpy()
    enhanced_spectrogram_wandb_image = wandb.Image(convert_spectrogram_to_heatmap(enhanced_spectrogram),
                                                   caption='PR')
    enhanced_wandb_audio = wandb.Audio(pr_signal.squeeze().numpy(), sample_rate=hr_sr, caption='PR')

    wandb_dict = {f'test samples/{filename}/lsd': lsd,
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


def create_wandb_table(args, data_loader, epoch):
    wandb_table = init_wandb_table()

    enumeratedDataloader = enumerate(data_loader)
    for i, data in enumeratedDataloader:
        if args.wandb.n_files_to_log_to_table and i >= args.wandb.n_files_to_log_to_table:
            break
        lr, hr, pr, filename = data
        filename = filename[0]
        lsd, visqol = run_metrics(hr, pr, args, filename)
        add_data_to_wandb_table((hr, lr, pr), (lsd, visqol), filename, args, wandb_table)

    wandb.log({"Results": wandb_table}, step=epoch)


def init_wandb_table():
    columns = ['filename', 'hr audio', 'hr spectogram', 'lr audio', 'lr spectogram', 'pr audio','pr spectogram',
               'lsd', 'visqol']
    table = wandb.Table(columns=columns)
    return table


def add_data_to_wandb_table(signals, metrics, filename, args, wandb_table):
    hr, lr, pr = signals

    spectrogram_transform = Spectrogram(n_fft=args.experiment.nfft)

    lr_upsampled = resample(lr, args.experiment.lr_sr, args.experiment.hr_sr)

    hr_spectrogram = spectrogram_transform(hr).log2()[0, :, :].numpy()
    lr_spectrogram = (SPECTOGRAM_EPSILON + spectrogram_transform(lr_upsampled)).log2()[0, :, :].numpy()
    pr_spectrogram = spectrogram_transform(pr).log2()[0, :, :].numpy()
    hr_wandb_spec = wandb.Image(convert_spectrogram_to_heatmap(hr_spectrogram))
    lr_wandb_spec = wandb.Image(convert_spectrogram_to_heatmap(lr_spectrogram))
    pr_wandb_spec = wandb.Image(convert_spectrogram_to_heatmap(pr_spectrogram))
    lsd, visqol = metrics

    hr_sr = args.experiment.hr_sr
    lr_sr = args.experiment.lr_sr

    hr_wandb_audio = wandb.Audio(hr.squeeze().numpy(), sample_rate=hr_sr, caption=filename + '_hr')
    lr_wandb_audio = wandb.Audio(lr.squeeze().numpy(), sample_rate=lr_sr, caption=filename + '_lr')
    pr_wandb_audio = wandb.Audio(pr.squeeze().numpy(), sample_rate=hr_sr, caption=filename + '_pr')

    wandb_table.add_data(filename, hr_wandb_audio, hr_wandb_spec, lr_wandb_audio, lr_wandb_spec,
                         pr_wandb_audio, pr_wandb_spec,
                         lsd, visqol)