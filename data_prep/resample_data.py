import sox
import os
import sys
import argparse
from multiprocessing import Pool


def resample_subdir(data_dir, data_subdir, out_dir, target_sr):
    print(f'resampling {data_subdir}')
    tfm = sox.Transformer()
    tfm.set_output_format(rate=target_sr)
    out_sub_dir = os.path.join(out_dir, data_subdir)
    if not os.path.isdir(out_sub_dir):
        os.makedirs(out_sub_dir)
    for file in os.listdir(os.path.join(data_dir, data_subdir)):
        out_path = os.path.join(out_sub_dir, file)
        in_path = os.path.join(data_dir, data_subdir, file)
        if os.path.isfile(out_path):
            print(f'{out_path} already exists.')
        elif not file.lower().endswith('.wav'):
            print(f'{in_path}: invalid file type.')
        else:
            success = tfm.build_file(input_filepath=in_path, output_filepath=out_path)
            if success:
                print(f'Succesfully saved {in_path} to {out_path}')


def resample_data(data_dir, out_dir, target_sr):
    with Pool() as p:
        p.starmap(resample_subdir,
                  [(data_dir, data_subdir, out_dir, target_sr) for data_subdir in os.listdir(data_dir)])


def parse_args():
    parser = argparse.ArgumentParser(description='Resample data.')
    parser.add_argument('--data_dir', help='directory containing source files')
    parser.add_argument('--out_dir', help='directory to write target files')
    parser.add_argument('--target_sr', type=int, help='target sample rate')
    return parser.parse_args()

"""Usage: python data_prep/resample_data.py --data_dir <path for source data> --out_dir <path for target data> --target_sr <target sample rate>"""
def main():
    args = parse_args()
    print(args)

    resample_data(args.data_dir, args.out_dir, args.target_sr)
    print(f'Done resampling to target rate {args.target_sr}.')


if __name__ == '__main__':
    main()