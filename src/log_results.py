import wandb
from torchaudio.functional import resample
from torchaudio.transforms import Spectrogram

import logging

from src.metrics import run_metrics
from src.utils import convert_spectrogram_to_heatmap

logger = logging.getLogger(__name__)

SPECTOGRAM_EPSILON = 1e-13


def create_wandb_table(args, data_loader, epoch):
    wandb_table = init_wandb_table()

    for i, data in enumerate(data_loader):
        lr, hr, pr, filename = data
        filename = filename[0]
        lsd, visqol = run_metrics(hr, pr, args, filename)
        add_data_to_wandb_table((hr, lr, pr), (lsd, visqol), filename, args, wandb_table)

    wandb.log({"Results": wandb_table}, step=epoch)


def log_results(args, dataloader, epoch):
    logger.info('logging results...')
    create_wandb_table(args, dataloader, epoch)


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
