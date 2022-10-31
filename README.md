# AERO
Audio Super Resolution in the Spectral Domain

## Preparing Data

### Resample Data

`python data_prep/resample_data.py --data_dir <path for 48 kHz data> --out_dir <path for 4 kHz data> --target_sr 4` \
`python data_prep/resample_data.py --data_dir <path for 48 kHz data> --out_dir <path for 16 kHz data> --target_sr 16` 

### Create egs files

`python data_prep/create_meta_files.py <path for 4 kHz data> egs/vctk/4-16 lr` \
`python data_prep/create_meta_files.py <path for 16 kHz data> egs/vctk/4-16 hr`

### Creating Dummy egs files (for debugging code)
If you want to create dummy egs files for debugging code on small number of samples.
(This might be a little buggy, make sure that the same files exist in high/low resolution meta files)

`python data_prep/create_meta_files.py <path for 4 kHz data> egs/vctk/4-16 l --n_samples_limit=32` \
`python data_prep/create_meta_files.py <path for 16 kHz data> egs/vctk/4-16 hr --n_samples_limit=32`
