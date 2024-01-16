import argparse,os
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='config file')
parser.add_argument('backbone', type=str, help='backbone name')
parser.add_argument('checkpoint', type=str, help='checkpoint file')
parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
parser.add_argument('--image_dir', type=str, default='images', help='image root')
parser.add_argument('--out_dir', type=str, default='det_mask', help='output root')
parser.add_argument('--device', type=str, default='cuda:0', help='device: gpu')
parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')

args = parser.parse_args()

model = init_detector(args.config, args.checkpoint, device=args.device)

img_files = os.listdir(args.image_dir)
img_paths = [f'{args.image_dir}/{p}' for p in img_files]

for n,p in zip(img_files,img_paths):
    file_name = n[:-4]
    out_file = p.replace(args.image_dir, args.out_dir).replace(file_name, f'{file_name}_{args.backbone}')
    result = inference_detector(model, p)
    model.show_result(
        p,
        result,
        score_thr=args.score_thr,
        show=False,
        bbox_color=args.palette,
        text_color=(200, 200, 200),
        mask_color=args.palette,
        out_file=out_file
    )
    print(f'{n}  {out_file}')