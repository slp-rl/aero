# AERO
**Audio Super Resolution in the Spectral Domain**

This is a modified version of the official PyTorch implemenation of *AERO: Audio Super Resolution in the Spectral Domain*: [paper](https://arxiv.org/abs/2211.12232), [project page](https://pages.cs.huji.ac.il/adiyoss-lab/aero/).

Checkpoint files are available! Details below.

## Requirements

Install requirements specified in `requirements.txt`:  
```pip install -r requirements.txt```

We ran our code on CUDA/12.4, we therefore installed pytorch/torchvision/torchaudio with the following:

```
pip install torch==2.4.1+cu124 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```
Torch 2.3.1 appears to have an autocast issue that prevents it from working with bfloat16, at least for this project.

Our code uses [hydra](https://hydra.cc/) to set parameters to different experiments.

### ViSQOL

If you want to run code without using ViSQOL, set `visqol: False` in file: `conf/main_config.yaml`.

In order to evaluate model output with the [ViSQOL](https://github.com/google/visqol) metric, one first needs to install 
Bazel and then ViSQOL.  
In our code, we use ViSQOL via its command line API by using a Python subprocess.

Build Bazel and ViSQOL following directions from [here](https://github.com/google/visqol#build).

Add the absolute path of the root directory of ViSQOL (where the WORKSPACE file is), to the `visqol path` parameter in 
`main_config.yaml`.

## Data

### Download data

For speech we use the [VCTK Corpus](https://datashare.ed.ac.uk/handle/10283/3443). \
For music we use the mixture tracks of [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav) dataset.
Make sure to download the uncompressed WAV version.

### Resample data

Data are a collection of high/low resolution pairs. Corresponding high and low resolution signals should be in
different folders.

In order to create each folder, one should run `resample_data` a total of 5 times,
to include all source/target pairs in both speech and music settings.

For speech, we use 4 lr-hr settings: 8-16 kHz, 8-24 kHz, 4-16 kHz, 12-48 kHz.
This requires to resample to 4 different resolutions (not including the original 48 kHz):
4, 8, 16, and 24 kHz.

For music, we downsample once to a target 11.025 kHz, from the original 44.1 kHz.

E.g. for 4 and 16 kHz: \
`python data_prep/resample_data.py --data_dir <path for 48 kHz data> --out_dir <path for 4 kHz data> --target_sr 4` \
`python data_prep/resample_data.py --data_dir <path for 48 kHz data> --out_dir <path for 16 kHz data> --target_sr 16` 

### Create egs files

For each low and high resolution pair, one should create "egs files" twice: for low and high resolution.  
`create_meta_files.py` creates a pair of train and val "egs files", each under its respective folder.
Each "egs file" contains meta information about the signals: paths and signal lengths.

e.g. to create egs files for the various speech settings:

`python data_prep/create_meta_files.py <path for 4 kHz data> egs/vctk/4-16 lr` \
`python data_prep/create_meta_files.py <path for 16 kHz data> egs/vctk/4-16 hr`

`python data_prep/create_meta_files.py <path for 8 kHz data> egs/vctk/8-16 lr` \
`python data_prep/create_meta_files.py <path for 16 kHz data> egs/vctk/8-16 hr`

`python data_prep/create_meta_files.py <path for 8 kHz data> egs/vctk/8-24 lr` \
`python data_prep/create_meta_files.py <path for 24 kHz data> egs/vctk/8-24 hr`

`python data_prep/create_meta_files.py <path for 12 kHz data> egs/vctk/12-48 lr` \
`python data_prep/create_meta_files.py <path for 46 kHz data> egs/vctk/12-48 hr`

### Creating dummy egs files (for debugging code)
If you want to create dummy egs files for debugging code on small number of samples.
(This might be a little buggy, make sure that the same files exist in high/low resolution meta (egs) files)

`python data_prep/create_meta_files.py <path for 4 kHz data> egs/vctk/4-16 lr --n_samples_limit=32` \
`python data_prep/create_meta_files.py <path for 16 kHz data> egs/vctk/4-16 hr --n_samples_limit=32`

## Train

Run `train.py` with `dset` and `experiment` parameters.  
(make sure that the parameters `lr_sr`, `hr_sr` in the experiment comply with the sample rates of the dataset). 

e.g. for upsampling from 4kHz to 16kHz, with `n_fft=512` and `hop_length=64`:
```
python train.py dset=4-16 experiment=aero_4-16_512_64
```

To train with multiple GPUs, run with parameter `ddp=true`. e.g.
```
python train.py dset=4-16 experiment=aero_4-16_512_64 ddp=true
```

## Test (on whole dataset)

- Make sure to create appropriate egs files for specific LR to HR setting
   - e.g. for `4-16`:  
       `python data_prep/create_meta_files.py <path for 4 kHz data> egs/vctk/4-16 lr` \
       `python data_prep/create_meta_files.py <path for 16 kHz data> egs/vctk/4-16 hr`
- Create a directory with experiment name in the format: `aero-nfft=<NFFT>-hl=<HOP_LENGTH>` (e.g. `aero-nfft=512-hl=64`)
- Copy/download appropriate `checkpoint.th` file to directory (make sure that the corresponding nfft,hop_length parameters correspond to experiment file)
- Run `python test.py dset=<LR>-<HR> experiment=aero_<LR>-<HR>_<NFFT>_<HOP_LENGTH>`

e.g. for upsampling from 4kHz to 16kHz, with `n_fft=512` and `hop_length=64`:

```
python test.py \
  dset=4-16 \
  experiment=aero_4-16_512_64
```

## Predict (on single sample)

- Copy/download appropriate `checkpoint.th` file to directory (make sure that the corresponding nfft,hop_length parameters
correspond to experiment file)
- Run predict.py with appending new `filename` and `output` parameters via hydra framework, corresponding to the input file and output directory respectively.

e.g. for upsampling from 4kHz to 16kHz, with `n_fft=512` and `hop_length=64`:

```
python predict.py \
  experiment=aero_4-16_512_64 \
  +filename=<absolute path to input file> \
  +output=<absolute path to output directory>
```

## Checkpoints

To use pre-trained models, one can download checkpoints
from [here](https://drive.google.com/drive/folders/1KuVJNkR7lZddvufmNsx-uAIluvb5XQ2L?usp=share_link).

Thank you for [@fmac2000](https://github.com/fmac2000) for providing checkpoints for the 16->48 kHz configuration! They are now included in the provided checkpoint folder.

## Additional models for this modified project

[AM Radio Super Resolution](https://www.mediafire.com/folder/orqm1wxjyikjp/test-v1)
[FM Radio Super Resolution](https://huggingface.co/sereich/FMRadioSR)

To link to checkpoint when testing or predicting, override/set path under `checkpoint_file:<path>`
in `conf/main_config.yaml.`  
e.g.

```
python test.py \
  dset=4-16 \
  experiment=aero_4-16_512_64 \
  +checkpoint_file=<path to appropriate checkpoint.th file>
```

Alternatively, make sure that the checkpoint file is in its corresponding output folder:  
For each low to high resolution setting, hydra creates a folder under `outputs/`: lr-hr (e.g. `outputs/4-16`), under
each such folder hydra creates a folder with the experiment name and n_fft and hop_length hyper-paremers (e.g.
`aero-nfft=512-hl=256`). Make sure that each checkpoint exists beforehand in appropriate output folder, if you download
the
[outputs](https://drive.google.com/drive/folders/1KuVJNkR7lZddvufmNsx-uAIluvb5XQ2L?usp=share_link) folder and place it
under the root directory (which contains `train.py` and `/src`), it should retain the appropriate structure and no
renaming should be necessary (make sure that `restart: false` in `conf/main_config.yaml`)
