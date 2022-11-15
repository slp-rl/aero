# AERO
Audio Super Resolution in the Spectral Domain

## Requirements

### ViSQOL

Build Bazel and ViSQOL following directions from [here](https://github.com/google/visqol#build).

Add the absolute path of the root directory of ViSQOL (where the WORKSPACE file is), to the `visqol path` parameter in 
`main_config.yaml`.
## Preparing Data

### Resample data

E.g. for 4 and 16 kHz: \
`python data_prep/resample_data.py --data_dir <path for 48 kHz data> --out_dir <path for 4 kHz data> --target_sr 4` \
`python data_prep/resample_data.py --data_dir <path for 48 kHz data> --out_dir <path for 16 kHz data> --target_sr 16` 

### Create egs files

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

```
python train.py dset=4-16 experiment=aero_4-16_512_64 \
```

## Test (on whole dataset)

- Make sure to create appropriate egs files for specific LR to HR setting
   - e.g. for `4-16`:  
       `python data_prep/create_meta_files.py <path for 4 kHz data> egs/vctk/4-16 lr` \
       `python data_prep/create_meta_files.py <path for 16 kHz data> egs/vctk/4-16 hr`
- Create a directory with experiment name in the format: `aero-nfft=<NFFT>-hl=<HOP_LENGTH>` (e.g. `aero-nfft=512-hl=64`)
- Copy/download appropriate `checkpoint.th` file to directory (make sure that the corresponding nfft,hop_length parameters correspond to experiment file)
- Run `python test.py dset=<LR>-<HR> experiment=aero_<LR>-<HR> _<NFFT>_<HOP_LENGTH>`

e.g. for upsampling from 4kHz to 16kHz:

`
python test.py dset=4-16 experiment=aero_4-16_512_64
`

## Predict (on single sample)