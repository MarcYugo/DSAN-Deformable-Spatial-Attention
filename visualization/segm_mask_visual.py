import argparse,os
from mmseg.apis import inference_model, init_model, show_result_pyplot
from mmseg.core.evaluation import get_palette
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='config file')
parser.add_argument('backbone', type=str, help='backbone name')
parser.add_argument('checkpoint', type=str, help='checkpoint file')
parser.add_argument('palette', type=str, help='palette')
parser.add_argument('--image_dir', type=str, default='images', help='image root')
parser.add_argument('--out_dir', type=str, default='segm_mask', help='output root')
parser.add_argument('--device', type=str, default='cuda:0', help='device: gpu')

args = parser.parse_args()

model = init_model(args.config, args.checkpoint, device=args.device)

img_files = os.listdir(args.image_dir)
img_paths = [f'{args.image_dir}/{p}' for p in img_files]

for n,p in zip(img_files,img_paths):
    file_name = n[:-4]
    out_file = p.replace(args.image_dir, args.out_dir).replace(file_name, f'{file_name}_{args.backbone}')
    result = inference_model(model, p)
    show_result_pyplot(model, p, result, get_palette(args), out_file=out_file)
    print(f'{n}  {out_file}')
