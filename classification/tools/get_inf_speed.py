import argparse
from models import *

import torch
from time import time

mods = {
    'dsan_t': dsan_t,
    'dsan_s': dsan_s,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help='model name')
    parser.add_argument('gpu_id',type=int,help='gpu id')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[2048, 1024],
        help='input image size')
    parser.add_argument('--mode', type=str, default='w/ head', help='with head or without head')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    torch.cuda.set_device(args.gpu_id)

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    model = mods[args.model_name]().cuda()
    model.eval()

    inputs = torch.randn(1, *input_shape).cuda()
    n = 1000
    if args.mode == 'w/ head':
        out = model(inputs)
        s = time()
        for i in range(n):
            out = model(inputs)
        e = time()
    elif args.mode == 'w/o head':
        out = model(inputs)
        s = time()
        for i in range(n):
            out = model.forward_backbone(inputs)
        e = time()
    print(f'{args.model_name} inference speed: {(e-s)/n} frames per second (input shape : {input_shape})')

if __name__ == '__main__':
    main()
