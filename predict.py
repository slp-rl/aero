import math
import os

import time

import hydra
import torch
import logging
from pathlib import Path

import torchaudio
from torchaudio.functional import resample

from src.enhance import write
from src.models import modelFactory
from src.model_serializer import SERIALIZE_KEY_MODELS, SERIALIZE_KEY_BEST_STATES, SERIALIZE_KEY_STATE
from src.utils import bold

logger = logging.getLogger(__name__)


SEGMENT_DURATION_SEC = 5
SEGMENT_OVERLAP_SAMPLES = 2048

def _load_model(args):
    model_name = args.experiment.model
    checkpoint_file = Path(args.checkpoint_file)
    model = modelFactory.get_model(args)['generator']
    package = torch.load(checkpoint_file, 'cpu')
    load_best = args.continue_best
    if  'state' in package.keys(): #raw model file
        logger.info(bold(f'Loading model {model_name} from file.'))
        model.load_state_dict(package[SERIALIZE_KEY_STATE])
    elif load_best:
        logger.info(bold(f'Loading model {model_name} from best state.'))
        model.load_state_dict(
            package[SERIALIZE_KEY_BEST_STATES]['generator'])
    else:
        logger.info(bold(f'Loading model {model_name} from last state.'))
        model.load_state_dict(package[SERIALIZE_KEY_MODELS]['generator'][SERIALIZE_KEY_STATE])

    return model

def crossfade_and_blend(out_clip, in_clip):
    fade_out = torchaudio.transforms.Fade(0,SEGMENT_OVERLAP_SAMPLES)
    fade_in = torchaudio.transforms.Fade(SEGMENT_OVERLAP_SAMPLES, 0)
    return fade_out(out_clip) + fade_in(in_clip)


@hydra.main(config_path="conf", config_name="main_config", version_base="1.1")  # for latest version of hydra=1.0
def main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)

    print(args)
    model = _load_model(args)
    device = torch.device('cuda')
    model.cuda()
    filename = args.filename
    file_basename = Path(filename).stem
    output_dir = args.output
    lr_sig, sr = torchaudio.load(str(filename))

    if args.experiment.upsample:
        lr_sig = resample(lr_sig, sr, args.experiment.hr_sr)
        sr = args.experiment.hr_sr

    logger.info(f'lr wav shape: {lr_sig.shape}')

    segment_duration_samples = sr * SEGMENT_DURATION_SEC
    n_chunks = math.ceil(lr_sig.shape[-1] / segment_duration_samples)
    logger.info(f'number of chunks: {n_chunks}')

    lr_chunks = []
    for i in range(n_chunks):
        start = i * segment_duration_samples
        end = min((i + 1) * segment_duration_samples, lr_sig.shape[-1])
        lr_chunks.append(lr_sig[:, start:end])

    pr_chunks = []

    model.eval()
    pred_start = time.time()
    with torch.no_grad():
        previous_chunk = None
        for i, lr_chunk in enumerate(lr_chunks):
            pr_chunk = None
            if previous_chunk is not None:
                combined_chunk = torch.cat((previous_chunk[...,-SEGMENT_OVERLAP_SAMPLES:], lr_chunk), 1)
                pr_combined_chunk = model(combined_chunk.unsqueeze(0).to(device)).squeeze(0)
                pr_chunk = pr_combined_chunk[...,SEGMENT_OVERLAP_SAMPLES:]
                pr_chunks[-1][...,-SEGMENT_OVERLAP_SAMPLES:] = crossfade_and_blend(pr_chunks[-1][...,-SEGMENT_OVERLAP_SAMPLES:], pr_combined_chunk.cpu()[...,:SEGMENT_OVERLAP_SAMPLES] )
            else:
                pr_chunk = model(lr_chunk.unsqueeze(0).to(device)).squeeze(0)
            logger.info(f'lr chunk {i} shape: {lr_chunk.shape}')
            logger.info(f'pr chunk {i} shape: {pr_chunk.shape}')
            pr_chunks.append(pr_chunk.cpu())
            previous_chunk = lr_chunk

    pred_duration = time.time() - pred_start
    logger.info(f'prediction duration: {pred_duration}')

    pr = torch.concat(pr_chunks, dim=-1)

    logger.info(f'pr wav shape: {pr.shape}')

    out_filename = os.path.join(output_dir, file_basename + '_pr.wav')
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f'saving to: {out_filename}, with sample_rate: {args.experiment.hr_sr}')

    write(pr, out_filename, args.experiment.hr_sr)

"""
Need to add filename and output to args.
Usage: python predict.py <experiment> +filename=<path to input file> +output=<path to output dir>
"""
if __name__ == "__main__":
    main()