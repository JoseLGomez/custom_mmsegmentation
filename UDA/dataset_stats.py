import argparse
import json
import os.path as osp

import mmcv
import numpy as np
from PIL import Image


def get_data(data_list):
    im_list = []
    with open(data_list,'r') as f:
        for line in f.readlines():
            im_list.append(line.rstrip().split(' ')[0])
    return im_list


def compute_class_stats(file):
    # re-assign labels to match the format of Cityscapes
    pil_label = Image.open(file)
    label = np.asarray(pil_label)
    sample_class_stats = {}
    for k in range(19):
        k_mask = label == k
        n = int(np.sum(k_mask))
        if n > 0:
            sample_class_stats[k] = n
    sample_class_stats['file'] = file
    return sample_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get dataset total pixels per class to use on Rare Class Sampling (RCS)')
    parser.add_argument('data_path', help='data path')
    parser.add_argument('--gt-file',default=None, help='Have priority over gt-dir')
    parser.add_argument('--gt-dir', default='labels', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=4, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    idx = 0
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(idx, file, n)]
            else:
                samples_with_class[c].append((idx, file, n))
        idx += 1
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    data_path = args.data_path
    out_dir = args.out_dir if args.out_dir else data_path
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(data_path, args.gt_dir)

    if args.gt_file is not None:
        poly_files = sorted(get_data(args.gt_file))
    else:
        poly_files = []
        for poly in mmcv.scandir(
                gt_dir, suffix=tuple(f'{i}.png' for i in range(10)),
                recursive=True):
            poly_file = osp.join(gt_dir, poly)
            poly_files.append(poly_file)
        poly_files = sorted(poly_files)

    only_postprocessing = False
    if not only_postprocessing:
        if args.nproc > 1:
            sample_class_stats = mmcv.track_parallel_progress(
                compute_class_stats, poly_files, args.nproc)
        else:
            sample_class_stats = mmcv.track_progress(compute_class_stats,
                                                     poly_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)


if __name__ == '__main__':
    main()
