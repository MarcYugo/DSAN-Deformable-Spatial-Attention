## Visualization
### CAM
Please make sure that scripts "cls_cam_visual.py" is placed alongside with model, the saved checkpoint, and its configuration file. This means you should copy this script into the "classification" folder when conducting the visualization experiment. you need to use the following terminal command to start the visualization:
```
python cls_cam_visual.py --model_name <dsan-t or dsan-s or ...> --images_dir <folder containing source images> --cams_dir <folder storing output visualization>
```
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
### Box and instance mask
Please make sure that scripts "det_box_ins_mask_visual.py" is placed alongside with model, the saved checkpoint, and its configuration file. This means you should copy this script into the "detection" folder when conducting the visualization experiment. you need to use the following terminal command to start the visualization:
```
python det_box_ins_mask_visual.py --config <corresponding configuration file of models> --backbone <dsan-t or dsan-s or ...> --checkpoint <checkpoint file of the corresponding models> --palette <color map styles of dataset> --image_dir <folder containing source images> --out_dir <folder storing output visualization>
```
