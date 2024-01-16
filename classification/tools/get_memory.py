import argparse
from models import *

import torch
from gpu_mem_track import MemTracker

mods = {
    'dsan_t': dsan_t,
    'dsan_s': dsan_s
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help='model name')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[2048, 1024],
        help='input image size')
    parser.add_argument('--gpu_id', type=int, default=0, help='device(gpu) id')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--mode', type=str, default='inference', help='inference or training')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    device = torch.device(f'cuda:{args.gpu_id}')
    torch.cuda.set_device(device)
    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')
    model = mods[args.model_name]()

    gpu_tracker = MemTracker()
    
    gpu_tracker.track()
    print('load data ..')
    inputs = torch.rand([args.batch_size] + list(input_shape)).cuda()
    gpu_tracker.track()
    print('load model ..')
    model = model.cuda()
    gpu_tracker.track()
    
    if args.mode == 'inference':
        print('training memory consumption')
        model.train()
        out = model(inputs)
        out.mean().backward()
        gpu_tracker.track()
    else:
        print('inference memory consumption')
        model.eval()
        out = model(inputs)
        gpu_tracker.track()

if __name__ == '__main__':
    main()
