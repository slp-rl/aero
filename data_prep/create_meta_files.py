import sox
import os
import sys
import argparse
import glob
import torchaudio
from collections import namedtuple
import json
from multiprocessing import Process, Manager
import pathlib

FILE_PATTERN='*_mic1.wav'
TOTAL_N_SPEAKERS=108
TRAIN_N_SPEAKERS=100
TEST_N_SPEAKERS=8

Info = namedtuple("Info", ["length", "sample_rate", "channels"])


def get_info(path):
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        # new version of torchaudio
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)


def add_subdir_meta(subdir_path, shared_meta, n_samples_limit):
    if n_samples_limit and len(shared_meta) > n_samples_limit:
        return
    print(f'creating meta for {subdir_path}')
    audio_files = glob.glob(os.path.join(subdir_path, FILE_PATTERN))
    for idx, file in enumerate(audio_files):
        info = get_info(file)
        shared_meta.append((file, info.length))


def create_subdirs_meta(subdirs_paths, n_samples_limit):
    with Manager() as manager:
        shared_meta = manager.list()
        processes = []
        for subdir_path in subdirs_paths:
            p = Process(target=add_subdir_meta, args=(subdir_path, shared_meta, n_samples_limit))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        meta = list(shared_meta)
        meta.sort()
        print(n_samples_limit)
        if n_samples_limit:
            meta = meta[:n_samples_limit]
        return meta

def create_meta(data_dir, n_samples_limit=None):
    root, subdirs, files = next(os.walk(data_dir, topdown=True))
    subdirs.sort()
    assert len(subdirs) == TOTAL_N_SPEAKERS
    train_subdirs_paths = [os.path.join(root, d) for d in subdirs[:TRAIN_N_SPEAKERS]]
    test_subdirs_paths = [os.path.join(root, d) for d in subdirs[TRAIN_N_SPEAKERS:]]
    assert len(test_subdirs_paths) == TEST_N_SPEAKERS
    train_meta = create_subdirs_meta(train_subdirs_paths, n_samples_limit)
    test_meta = create_subdirs_meta(test_subdirs_paths, n_samples_limit)

    if n_samples_limit:
        assert len(train_meta) == n_samples_limit
        assert len(test_meta) == n_samples_limit

    return train_meta, test_meta



def parse_args():
    parser = argparse.ArgumentParser(description='Resample data.')
    parser.add_argument('data_dir', help='directory containing source files')
    parser.add_argument('target_dir', help='output directory for created json files')
    parser.add_argument('json_filename', help='filename for created json files')
    parser.add_argument('--n_samples_limit', type=int, help='limit number of files')
    return parser.parse_args()



"""
usage: python data_prep/create_meta_file.py <data_dir_path> <target_dir> <json_filename>
"""
def main():
    args = parse_args()
    print(args)

    os.makedirs(args.target_dir, exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'tr'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'val'), exist_ok=True)


    train_meta, test_meta = create_meta(args.data_dir, args.n_samples_limit)

    train_json_object = json.dumps(train_meta, indent=4)
    test_json_object = json.dumps(test_meta, indent=4)
    with open(os.path.join(args.target_dir, 'tr', args.json_filename + '.json'), "w") as train_out:
        train_out.write(train_json_object)
    with open(os.path.join(args.target_dir, 'val', args.json_filename + '.json'), "w") as test_out:
        test_out.write(test_json_object)

    print(f'Done creating meta for {args.data_dir}.')


if __name__ == '__main__':
    main()