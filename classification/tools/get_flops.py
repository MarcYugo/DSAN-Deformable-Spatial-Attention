# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from mmcv.cnn import get_model_complexity_info
from models import *

import torch
from torchprofile import profile_macs

mods = {
    'dsan_t': dsan_t,
    'dsan_s': dsan_s,
}

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('model_name', help='model name')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[2048, 1024],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    model = mods[args.model_name]().cuda()
    model.eval()

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))

    inputs = torch.randn(1, *input_shape).cuda()
    macs = profile_macs(model, inputs) / 1e9
    print(f'GFLOPs {macs}.')


if __name__ == '__main__':
    main()
