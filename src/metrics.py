import errno
import os
import subprocess
import logging
import time

import numpy as np
import sox
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

SLEEP_DURATION = 0.1
VISQOL_MIN_DURATION = 0.48

def run_metrics(clean, estimate, args, filename):
    hr_sr = args.experiment.hr_sr if 'experiment' in args else args.hr_sr
    speech_mode = args.experiment.speech_mode if 'speech_mode' in args.experiment else True
    lsd, visqol = get_metrics(clean, estimate, hr_sr, filename, speech_mode,args)
    return lsd, visqol


def get_metrics(clean, estimate, sr, filename, speech_mode, args):
    calc_visqol = args.visqol and args.visqol_path
    visqol_path = args.visqol_path
    clean = clean.squeeze(dim=1)
    estimate = estimate.squeeze(dim=1)
    estimate_numpy = estimate.numpy()
    clean_numpy = clean.numpy()

    lsd = get_lsd(clean, estimate).item()
    visqol = get_visqol(clean_numpy, estimate_numpy, filename, sr, speech_mode, visqol_path) if calc_visqol else 0
    return lsd, visqol


class STFTMag(nn.Module):
    def __init__(self,
                 nfft=1024,
                 hop=256):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)

    # x: [B,T] or [T]
    @torch.no_grad()
    def forward(self, x):
        T = x.shape[-1]
        stft = torch.stft(x,
                          self.nfft,
                          self.hop,
                          window=self.window,
                          )  # return_complex=False)  #[B, F, TT,2]
        mag = torch.norm(stft, p=2, dim=-1)  # [B, F, TT]
        return mag

# taken from: https://github.com/nanahou/metric/blob/master/measure_SNR_LSD.py
def get_lsd(ref_sig, out_sig):
    """
       Compute LSD (log spectral distance)
       Arguments:
           out_sig: vector (torch.Tensor), enhanced signal [B,T]
           ref_sig: vector (torch.Tensor), reference signal(ground truth) [B,T]
    """

    stft = STFTMag(2048, 512)
    sp = torch.log10(stft(ref_sig).square().clamp(1e-8))
    st = torch.log10(stft(out_sig).square().clamp(1e-8))
    return (sp - st).square().mean(dim=1).sqrt().mean()


# based on: https://github.com/eagomez2/upf-smc-speech-enhancement-thesis/blob/main/src/utils/evaluation_process.py
def get_visqol(ref_sig, out_sig, filename, sr, speech_mode, visqol_path):
    tmp_reference = f"{filename}_ref.wav"
    tmp_estimation = f"{filename}_est.wav"

    reference_abs_path = os.path.abspath(tmp_reference)
    estimation_abs_path = os.path.abspath(tmp_estimation)

    if speech_mode:
        target_sr = 16000 if sr != 16000 else None
    else:
        target_sr = 48000 if sr != 48000 else None

    tfm = sox.Transformer()
    tfm.convert(bitdepth=16, samplerate=target_sr)
    ref_sig = np.transpose(ref_sig)
    out_sig = np.transpose(out_sig)

    try:
        tfm.build_file(input_array=ref_sig, sample_rate_in=sr, output_filepath=reference_abs_path)
        tfm.build_file(input_array=out_sig, sample_rate_in=sr, output_filepath=estimation_abs_path)
        while not os.path.exists(reference_abs_path) and not os.path.exists(estimation_abs_path):
            time.sleep(SLEEP_DURATION)

        if not os.path.isfile(reference_abs_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), reference_abs_path)
        if not os.path.isfile(estimation_abs_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), estimation_abs_path)

        ref_duration = sox.file_info.duration(reference_abs_path)
        est_duration = sox.file_info.duration(estimation_abs_path)

        if ref_duration < VISQOL_MIN_DURATION or est_duration < VISQOL_MIN_DURATION:
            raise ValueError('File duration is too small.')

        visqol_cmd = ("cd " + visqol_path + "; " +
                      "./bazel-bin/visqol "
                      f"--reference_file {reference_abs_path} "
                      f"--degraded_file {estimation_abs_path} ")

        if speech_mode:
            visqol_cmd += f"--use_speech_mode"

        visqol = subprocess.run(visqol_cmd, shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # parse stdout to get the current float value
        visqol = visqol.stdout.decode("utf-8").split("\t")[-1].replace("\n", "")
        visqol = float(visqol)

    except FileNotFoundError as e:
        logger.info(f'visqol: failed to create {filename}')
        logger.info(str(e))
        visqol = 0

    except Exception as e:
        logger.info(f'failed to get visqol of {filename}')
        logger.info(str(e))
        visqol = 0

    else:
        # remove files to avoid filling space storage
        os.remove(reference_abs_path)
        os.remove(estimation_abs_path)

    return visqol
