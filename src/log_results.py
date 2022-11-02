import math
import os
import pandas as pd
import wandb
from torchaudio.functional import resample
from torchaudio.transforms import Spectrogram
import numpy as np
from torchaudio import transforms

from .metrics import run_metrics, get_snr

from src.utils import match_signal

import logging

from .utils import convert_spectrogram_to_heatmap

logger = logging.getLogger(__name__)

SPECTOGRAM_EPSILON = 1e-13

RESULTS_DF_FILENAME = 'filename'
RESULTS_DF_LR_SNR = 'lr snr'
RESULTS_DF_PR_SNR = 'pr snr'
RESULTS_DF_PESQ = 'pesq'
RESULTS_DF_STOI = 'stoi'
RESULTS_DF_LSD = 'lsd'
RESULTS_DF_SISNR = 'sisnr'
RESULTS_DF_VISQOL = 'visqol'

HISTOGRAM_DF_RANGE = 'range'
HISTOGRAM_DF_N_SAMPLES = 'n_samples'
HISTOGRAM_DF_AVG_PESQ = 'avg pesq'
HISTOGRAM_DF_AVG_STOI = 'avg stoi'
HISTOGRAM_DF_AVG_LSD = 'avg lsd'
HISTOGRAM_DF_AVG_SISNR = 'avg sisnr'
HISTOGRAM_DF_AVG_VISQOL = 'avg visqol'

DEFAULT_HISTOGRAM_MAX_SNR = 40.0

WANDB_CUSTOM_CHART_NAME = "huji-dl-audio-lab/non-sorted-bar-chart"


def create_wandb_table(args, data_loader, epoch):
    wandb_table = init_wandb_table()

    for i, data in enumerate(data_loader):
        lr, hr, pr, filename = data
        filename = filename[0]
        lsd, visqol = run_metrics(hr, pr, args, filename)
        add_data_to_wandb_table((hr, lr, pr), (lsd, visqol), filename, args, wandb_table)

    wandb.log({"Results": wandb_table}, step=epoch)


# def get_histogram_intervals(max_snr_value, n_bins):
#     step_size = DEFAULT_HISTOGRAM_MAX_SNR / n_bins
#     max_snr_value = max(max_snr_value, DEFAULT_HISTOGRAM_MAX_SNR)
#     intervals = np.arange(max_snr_value + 1e-3, step=step_size)
#     while intervals[-1] < max_snr_value:
#         intervals = np.append(intervals, intervals[-1] + step_size)
#     return intervals


# def create_results_histogram_df(results_df, n_bins, epoch):
#     results_histogram_df = pd.DataFrame(columns=[HISTOGRAM_DF_RANGE, HISTOGRAM_DF_N_SAMPLES, HISTOGRAM_DF_AVG_PESQ,
#                                                  HISTOGRAM_DF_AVG_STOI, HISTOGRAM_DF_AVG_LSD, HISTOGRAM_DF_AVG_SISNR,
#                                                  HISTOGRAM_DF_AVG_VISQOL])
#     lr_snr_values = results_df[RESULTS_DF_LR_SNR]
#     bin_indices, bins = pd.cut(lr_snr_values, get_histogram_intervals(lr_snr_values.max(), n_bins),
#                                labels=False, retbins=True, right=False)
#     wandb_ranges = []
#     n_samples_per_bin = []
#     wandb_pesq = []
#     wandb_stoi = []
#     wandb_lsd = []
#     wandb_sisnr = []
#     wandb_visqol = []
#     total_n_samples = 0
#     for i in range(len(bins) - 1):
#         bin_range = (float("{:.2f}".format(bins[i])), float("{:.2f}".format(bins[i + 1])))
#         wandb_ranges.append(', '.join(str(x) for x in bin_range))
#         n_samples_per_bin_i = len(bin_indices[bin_indices == i])
#         n_samples_per_bin.append(n_samples_per_bin_i)
#         total_n_samples += n_samples_per_bin_i
#         bin_avg_pesq = results_df.pesq[bin_indices == i].mean()
#         bin_avg_stoi = results_df.stoi[bin_indices == i].mean()
#         bin_avg_lsd = results_df.lsd[bin_indices == i].mean()
#         bin_avg_sisnr = results_df.sisnr[bin_indices == i].mean()
#         bin_avg_visqol = results_df.visqol[bin_indices == i].mean()
#         bin_avg_pesq = 0 if math.isnan(bin_avg_pesq) else bin_avg_pesq
#         bin_avg_stoi = 0 if math.isnan(bin_avg_stoi) else bin_avg_stoi
#         bin_avg_lsd = 0 if math.isnan(bin_avg_lsd) else bin_avg_lsd
#         bin_avg_sisnr = 0 if math.isnan(bin_avg_sisnr) else bin_avg_sisnr
#         bin_avg_visqol = 0 if math.isnan(bin_avg_visqol) else bin_avg_visqol
#         wandb_pesq.append(bin_avg_pesq)
#         wandb_stoi.append(bin_avg_stoi)
#         wandb_lsd.append(bin_avg_lsd)
#         wandb_sisnr.append(bin_avg_sisnr)
#         wandb_visqol.append(bin_avg_visqol)
#         results_histogram_df.loc[i] = [bin_range, n_samples_per_bin_i, bin_avg_pesq, bin_avg_stoi, bin_avg_lsd,
#                                        bin_avg_sisnr, bin_avg_visqol]
#     log_wandb_bar_chart([[wandb_range, pesq, n_samples_per_bin_i] for (wandb_range, pesq, n_samples_per_bin_i)
#                          in zip(wandb_ranges, wandb_pesq, n_samples_per_bin)],
#                         ['ranges', 'pesq', 'n_samples_per_bin_i'], 'pesq', 'Average PESQ per SNR range', epoch)
#     log_wandb_bar_chart([[wandb_range, stoi, n_samples_per_bin_i] for (wandb_range, stoi, n_samples_per_bin_i)
#                          in zip(wandb_ranges, wandb_stoi, n_samples_per_bin)],
#                         ['ranges', 'stoi', 'n_samples_per_bin_i'], 'stoi', 'Average STOI per SNR range', epoch)
#     log_wandb_bar_chart([[wandb_range, lsd, n_samples_per_bin_i] for (wandb_range, lsd, n_samples_per_bin_i)
#                          in zip(wandb_ranges, wandb_lsd, n_samples_per_bin)],
#                         ['ranges', 'lsd', 'n_samples_per_bin_i'], 'lsd', 'Average LSD per SNR range', epoch)
#     log_wandb_bar_chart([[wandb_range, sisnr, n_samples_per_bin_i] for (wandb_range, sisnr, n_samples_per_bin_i)
#                          in zip(wandb_ranges, wandb_sisnr, n_samples_per_bin)],
#                         ['ranges', 'sisnr', 'n_samples_per_bin_i'], 'sisnr', 'Average SISNR per SNR range', epoch)
#     log_wandb_bar_chart([[wandb_range, visqol, n_samples_per_bin_i] for (wandb_range, visqol, n_samples_per_bin_i)
#                          in zip(wandb_ranges, wandb_visqol, n_samples_per_bin)],
#                         ['ranges', 'visqol', 'n_samples_per_bin_i'], 'visqol', 'Average VISQOL per SNR range', epoch)
#     return results_histogram_df


# def log_wandb_bar_chart(data, column_names, table_name, title, epoch):
#     table = wandb.Table(data=data, columns=column_names)
#     fields = {"label": column_names[0], "value": column_names[1]}
#     custom_chart = wandb.plot_table(WANDB_CUSTOM_CHART_NAME, table, fields, {"title": title})
#     wandb.log({table_name: custom_chart}, step=epoch)


def log_results(args, dataloader, epoch):
    logger.info('logging results...')
    # results_out_path = 'results.csv'
    # if os.path.isfile(results_out_path):
    #     logger.info('results.csv file already exists. Overwriting...')
        # results_df = pd.read_csv(results_out_path, index_col=False)
    # else:
    #     results_df = create_results_df(args, dataloader, epoch)
    #     results_df.to_csv(results_out_path)
    create_wandb_table(args, dataloader, epoch)
    #
    # n_bins = args.n_bins
    # histogram_out_path = 'results_histogram_' + str(n_bins) + '.csv'
    # if os.path.isfile(histogram_out_path):
    #     logger.info('histogram file already exists. Overwriting...')
    # results_histogram_df = create_results_histogram_df(results_df, n_bins, epoch)
    # results_histogram_df.to_csv(histogram_out_path)

    # if not os.path.isfile(histogram_out_path):
    #     results_histogram_df = create_results_histogram_df(results_df, n_bins, epoch)
    #     results_histogram_df.to_csv(histogram_out_path)
    # else:
    #     logger.info('histogram file already exists.')


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
