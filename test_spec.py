import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
from torch.nn import functional as F
from torchaudio.functional import resample

from src.models.spec import spectro, ispectro

import matplotlib
import matplotlib.pyplot as plt


def scale_minmax(X, min=0.0, max=1.0):
    isnan = np.isnan(X).any()
    isinf = np.isinf(X).any()
    if isinf:
        X[X == np.inf] = 1e9
        X[X == -np.inf] = 1e-9
    if isnan:
        X[X == np.nan] = 1e-9
    # logger.info(f'isnan: {isnan}, isinf: {isinf}, max: {X.max()}, min: {X.min()}')

    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def spectro(x, n_fft=512, hop_length=None, pad=0, win_length=None):
    *other, length = x.shape
    x = x.reshape(-1, length)
    z = torch.stft(x,
                   n_fft * (1 + pad),
                   hop_length or n_fft // 4,
                   window=None,
                   win_length=win_length or n_fft,
                   normalized=True,
                   center=True,
                   return_complex=True,
                   pad_mode='reflect')
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)


def convert_spectrogram_to_heatmap(spectrogram):
    spectrogram += 1e-9
    spectrogram = scale_minmax(spectrogram, 0, 255).astype(np.uint8).squeeze()
    spectrogram = np.flip(spectrogram, axis=0)
    spectrogram = 255 - spectrogram
    # spectrogram = (255 * (spectrogram - np.min(spectrogram)) / np.ptp(spectrogram)).astype(np.uint8).squeeze()[::-1,:]
    heatmap = cv2.applyColorMap(spectrogram, cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('lr_path', type=str)
    parser.add_argument('hr_path', type=str)

    return parser


def main(args, scale=4):
    lr_signal_path = args.lr_path
    lr_filename = Path(lr_signal_path).stem
    lr_signal, sr = torchaudio.load(lr_signal_path)

    lr_signal = lr_signal.mean(axis=0)

    hr_signal_path = args.hr_path
    hr_filename = Path(hr_signal_path).stem
    hr_signal, sr = torchaudio.load(hr_signal_path)

    hr_signal = hr_signal.mean(axis=0)

    n_fft = 512
    win_length = n_fft // (scale)
    hop_length = n_fft // (4 * scale)

    new_win_length = win_length * scale
    new_hop_length = hop_length * scale

    new_sr = sr * scale

    if np.mod(lr_signal.shape[-1], hop_length):
        lr_signal_padded = F.pad(lr_signal, (0, hop_length - np.mod(lr_signal.shape[-1], hop_length)))
    else:
        lr_signal_padded = lr_signal

    if np.mod(hr_signal.shape[-1], new_hop_length):
        hr_signal_padded = F.pad(hr_signal, (0, new_hop_length - np.mod(hr_signal.shape[-1], new_hop_length)))
    else:
        hr_signal_padded = hr_signal

    print(f'signal_padded shape: {lr_signal_padded.shape}')

    signal_upsampled = resample(lr_signal, sr / 2, new_sr / 2)

    print(f'signal_upsampled shape: {signal_upsampled.shape}')

    print(f'n_fft: {n_fft}, hop_length: {hop_length}, win_length: {win_length}')
    print(f'new n_fft: {n_fft}, hop_length: {new_hop_length}, win_length: {new_win_length}')

    sig_spectro = spectro(lr_signal_padded, n_fft=n_fft, hop_length=new_hop_length,
                          win_length=new_win_length)
    plt.imshow(convert_spectrogram_to_heatmap(sig_spectro.abs().pow(2).log2().numpy()))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(lr_filename + ' signal' + f' hl: {new_hop_length}, win:{new_win_length}, fft: {n_fft}')

    plt.show()

    sig_spectro = spectro(lr_signal_padded, n_fft=n_fft, hop_length=hop_length,
                                       win_length=win_length)
    plt.imshow(convert_spectrogram_to_heatmap(sig_spectro.abs().pow(2).log2().numpy()))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(lr_filename + ' signal' + f' hl: {hop_length}, win:{win_length}, fft: {n_fft}')

    plt.show()

    if np.mod(signal_upsampled.shape[-1], hop_length):
        lr_signal_upsampled_padded_1 = F.pad(signal_upsampled,
                                           (0, hop_length - np.mod(signal_upsampled.shape[-1], hop_length)))
    else:
        lr_signal_upsampled_padded_1 = signal_upsampled

    print(f'lr_signal_upsampled_padded_1 shape: {lr_signal_upsampled_padded_1.shape}')

    sig_upsampled_spectro = spectro(lr_signal_upsampled_padded_1, n_fft=n_fft, hop_length=hop_length,
                                    win_length=win_length)
    plt.imshow(convert_spectrogram_to_heatmap(sig_upsampled_spectro.abs().pow(2).log2().numpy()))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(lr_filename + ' signal upsampled' + f' hl: {hop_length}, win:{win_length}, fft: {n_fft}')

    plt.show()

    if np.mod(signal_upsampled.shape[-1], new_hop_length):
        lr_signal_upsampled_padded_2 = F.pad(signal_upsampled,
                                           (0, new_hop_length - np.mod(signal_upsampled.shape[-1], new_hop_length)))
    else:
        lr_signal_upsampled_padded_2 = signal_upsampled

    print(f'lr_signal_upsampled_padded_2 shape: {lr_signal_upsampled_padded_2.shape}')

    sig_upsampled_spectro = spectro(lr_signal_upsampled_padded_2, n_fft=n_fft, hop_length=new_hop_length,
                                    win_length=new_win_length)
    plt.imshow(convert_spectrogram_to_heatmap(sig_upsampled_spectro.abs().pow(2).log2().numpy()))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(lr_filename + ' signal upsampled' + f' hl: {new_hop_length}, win:{new_win_length}, fft: {n_fft}')

    plt.show()

    sig_upsampled_spectro = spectro(hr_signal_padded, n_fft=n_fft, hop_length=new_hop_length,
                                    win_length=new_win_length)
    plt.imshow(convert_spectrogram_to_heatmap(sig_upsampled_spectro.abs().pow(2).log2().numpy()))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(hr_filename + ' signal' + f' hl: {new_hop_length}, win:{new_win_length}, fft: {n_fft}')

    plt.show()




if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)