import os
import logging
import PIL
import torch

from src.ddp import distrib
from src.data.datasets import match_signal
from src.enhance import save_wavs, save_specs
from src.metrics import run_metrics
from src.utils import LogProgress, bold
from src.wandb_logger import log_data_to_wandb

logger = logging.getLogger(__name__)



def evaluate_lr_hr_pr_data(data, wandb_n_files_to_log, files_to_log, epoch, args):
    lr, hr, pr, filename = data
    filename = filename[0]
    hr_sr = args.experiment.hr_sr if 'experiment' in args else args.hr_sr
    if args.experiment.upsample:
        lr_sr = hr_sr
    else:
        lr_sr = args.experiment.lr_sr if 'experiment' in args else args.lr_sr

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

    lsd_i, visqol_i = run_metrics(hr, pr, args, filename)
    if filename in files_to_log:
        log_data_to_wandb(pr, hr, lr, lsd_i, visqol_i,
                          filename, epoch, lr_sr, hr_sr, lr_spec, pr_spec, hr_spec)

    return {'lsd': lsd_i, 'visqol': visqol_i, 'filename': filename}

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
    model.eval()
    with torch.no_grad():
        pr, pr_spec, lr_spec = model(lr, return_spec=True, return_lr_spec=True)
    model.train()
    pr = match_signal(pr, hr.shape[-1])
    hr_spec = model._spec(hr, scale=True)
    filename = Path(hr_path[0]).stem

    if wandb_n_files_to_log == -1 or len(files_to_log) < wandb_n_files_to_log:
        files_to_log.append(filename)

    if args.device != 'cpu':
        hr = hr.cpu()
        pr = pr.cpu()
        lr = lr.cpu()

    lsd_i, visqol_i = run_metrics(hr, pr, args, filename)
    if filename in files_to_log:
        log_data_to_wandb(pr, hr, lr, lsd_i, visqol_i,
                          filename, epoch, lr_sr, hr_sr, lr_spec.cpu(), pr_spec.cpu(), hr_spec.cpu())

    if enhance:
        os.makedirs(args.samples_dir, exist_ok=True)
        lr_sr = args.experiment.hr_sr if args.experiment.upsample else args.experiment.lr_sr
        save_wavs(pr, lr, hr, [os.path.join(args.samples_dir, filename)], lr_sr, args.experiment.hr_sr)
        save_specs(lr_spec, pr_spec, hr_spec, os.path.join(args.samples_dir, filename))

    return {'lsd': lsd_i, 'visqol': visqol_i, 'filename': filename}


def evaluate_on_saved_data(args, data_loader, epoch):

    total_lsd = 0
    total_visqol = 0

    lsd_count = 0
    visqol_count = 0

    total_cnt = 0

    files_to_log = []
    wandb_n_files_to_log = args.wandb.n_files_to_log if 'wandb' in args else args.wandb_n_files_to_log

    with torch.no_grad():
        iterator = LogProgress(logger, data_loader, name="Eval estimates")
        for i, data in enumerate(iterator):
            metrics_i = evaluate_lr_hr_pr_data(data, wandb_n_files_to_log, files_to_log, epoch, args)

            total_lsd += metrics_i['lsd']
            total_visqol += metrics_i['visqol']

            lsd_count += 1 if metrics_i['lsd'] != 0 else 0
            visqol_count += 1 if metrics_i['visqol'] != 0 else 0

            total_cnt += 1

    if lsd_count != 0:
        avg_lsd, = [total_lsd / lsd_count]
    else:
        avg_lsd = 0

    if visqol_count != 0:
        avg_visqol, = [total_visqol / visqol_count]
    else:
        avg_visqol = 0

    logger.info(bold(
        f'{args.experiment.name}, {args.experiment.lr_sr}->{args.experiment.hr_sr}. Test set performance:'
        f'LSD={avg_lsd} ({lsd_count}/{total_cnt}), VISQOL={avg_visqol} ({visqol_count}/{total_cnt}).'))

    return avg_lsd, avg_visqol


def evaluate(args, data_loader, epoch, model):
    total_lsd = 0
    total_visqol = 0

    lsd_count = 0
    visqol_count = 0

    total_cnt = 0

    total_filenames = []

    files_to_log = []
    wandb_n_files_to_log = args.wandb.n_files_to_log if 'wandb' in args else args.wandb_n_files_to_log

    with torch.no_grad():
        iterator = LogProgress(logger, data_loader, name="Eval estimates")
        for i, data in enumerate(iterator):

            metrics_i = evaluate_lr_hr_data(data, model, wandb_n_files_to_log, files_to_log, epoch, args)
            total_lsd += metrics_i['lsd']
            total_visqol += metrics_i['visqol']

            total_filenames.append(metrics_i['filename'])

            lsd_count += 1 if metrics_i['lsd'] != 0 else 0
            visqol_count += 1 if metrics_i['visqol'] != 0 else 0

            total_cnt += 1

    if lsd_count != 0:
        avg_lsd, = distrib.average([total_lsd / lsd_count], lsd_count)
    else:
        avg_lsd = 0
    if visqol_count != 0:
        avg_visqol, = distrib.average([total_visqol / visqol_count], visqol_count)
    else:
        avg_visqol = 0


    logger.info(bold(
        f'{args.experiment.name}, {args.experiment.lr_sr}->{args.experiment.hr_sr}. Test set performance:'
                                f'LSD={avg_lsd} ({lsd_count}/{total_cnt}), VISQOL={avg_visqol} ({visqol_count}/{total_cnt}).'))
    return avg_lsd, avg_visqol, total_filenames
