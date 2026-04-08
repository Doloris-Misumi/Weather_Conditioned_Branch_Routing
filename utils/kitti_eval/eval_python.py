import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import kitti_common as kitti
from eval import get_official_eval_result
import numpy as np

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--header',
        required=True,
        help='Path to the directory containing pred/, gt/, and val.txt.',
    )
    args = parser.parse_args()

    path_header = args.header
    det_path = path_header + "/pred"
    dt_annos = kitti.get_label_annos(det_path)
    gt_path = path_header + "/gt"
    gt_split_file = path_header + "/val.txt" # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
    val_image_ids = _read_imageset_file(gt_split_file)
    gt_annos = kitti.get_label_annos(gt_path, val_image_ids)

    dict_metrics, eval_sed = get_official_eval_result(gt_annos, dt_annos, 0, iou_mode='easy', is_return_with_dict=True)
    print(eval_sed)
    print(dict_metrics)
