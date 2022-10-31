import sox
import os
import sys
import argparse
from multiprocessing import Pool


def create_speakers_file(sr):
    pass


def create_recordings_file(sr):
    pass



def parse_args():
    parser = argparse.ArgumentParser(description='Prepare data data.')
    parser.add_argument('--data_dir', help='directory containing source files')
    parser.add_argument('--out_dir', help='directory to write target files')
    parser.add_argument('--source_sr', type=int, help='source sample rate')
    parser.add_argument('--target_sr', type=int, help='target sample rate')
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    resample_data(args.data_dir, args.out_dir, args.target_sr)
    print(f'Done resampling to target rate {args.target_sr}.')


if __name__ == '__main__':
    main()