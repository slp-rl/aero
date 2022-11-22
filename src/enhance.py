import os
import logging

import torch
import torchaudio
from PIL import Image

from src.utils import LogProgress, convert_spectrogram_to_heatmap
logger = logging.getLogger(__name__)

def get_estimate(model, lr_sig):
    torch.set_num_threads(1)
    with torch.no_grad():
        out = model(lr_sig)
    return out


def write(wav, filename, sr):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)


def save_wavs(processed_sigs, lr_sigs, hr_sigs, filenames, lr_sr, hr_sr):
    # Write result
    for lr, hr, pr, filename in zip(lr_sigs, hr_sigs, processed_sigs, filenames):
        write(lr, filename + "_lr.wav", sr=lr_sr)
        write(hr, filename + "_hr.wav", sr=hr_sr)
        write(pr, filename + "_pr.wav", sr=hr_sr)

def save_specs(lr_spec, pr_spec, hr_spec, filename):
    lr_spec_path = filename + "_lr_spec.png"
    if not os.path.isfile(lr_spec_path):
        lr_spec = lr_spec.cpu().abs().pow(2).log2()[0, :, :].numpy()
        lr_spec = convert_spectrogram_to_heatmap(lr_spec)
        lr_spec_img = Image.fromarray(lr_spec)
        lr_spec_img.save(lr_spec_path)

    hr_spec_path = filename + "_hr_spec.png"
    if not os.path.isfile(hr_spec_path):
        hr_spec = hr_spec.cpu().abs().pow(2).log2()[0, :, :].numpy()
        hr_spec = convert_spectrogram_to_heatmap(hr_spec)
        hr_spec_img = Image.fromarray(hr_spec)
        hr_spec_img.save(hr_spec_path)

    pr_spec = pr_spec.cpu().abs().pow(2).log2()[0, :, :].numpy()
    pr_spec = convert_spectrogram_to_heatmap(pr_spec)
    pr_spec_img = Image.fromarray(pr_spec)
    pr_spec_img.save(filename + "_pr_spec.png")


def enhance(dataloader, model, args):
    model.eval()

    os.makedirs(args.samples_dir, exist_ok=True)
    lr_sr = args.experiment.lr_sr if 'experiment' in args else args.lr_sr
    hr_sr = args.experiment.hr_sr if 'experiment' in args else args.hr_sr

    total_filenames = []

    iterator = LogProgress(logger, dataloader, name="Generate enhanced files")

    for i, data in enumerate(iterator):
        # Get batch data
        (lr_sigs, lr_paths), (hr_sigs, hr_paths) = data
        lr_sigs = lr_sigs.to(args.device)
        hr_sigs = hr_sigs.to(args.device)
        filenames = [os.path.join(args.samples_dir, os.path.basename(path).rsplit(".", 1)[0]) for path in lr_paths]
        total_filenames += [os.path.basename(path).rsplit(".", 1)[0] for path in lr_paths]

        estimates = get_estimate(model, lr_sigs)
        save_wavs(estimates, lr_sigs, hr_sigs, filenames, lr_sr, hr_sr)

        if i == args.enhance_samples_limit:
            break
    model.train()
    return total_filenames
