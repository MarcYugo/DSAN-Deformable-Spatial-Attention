## Visualization
### CAM
Please make sure that scripts "cls_cam_visual.py" is placed alongside with model, the saved checkpoint, and its configuration file. This means you should copy this script into the "classification" folder when conducting the visualization experiment. you need to use the following terminal command to start the visualization:
```
python cls_cam_visual.py --model_name <dsan-t or dsan-s or ...> --images_dir <folder containing source images> --cams_dir <folder storing output visualization>
```

### Box and instance segmentation mask
Please make sure that scripts "det_box_ins_mask_visual.py" is placed alongside with model, the saved checkpoint, and its configuration file. This means you should copy this script into the "detection" folder when conducting the visualization experiment. you need to use the following terminal command to start the visualization:
```
python det_box_ins_mask_visual.py --config <corresponding configuration file of models> --backbone <dsan-t or dsan-s or ...> --checkpoint <checkpoint file of the corresponding models> --palette <color map styles of dataset> --image_dir <folder containing source images> --out_dir <folder storing output visualization>
```

### Semantic segmentation mask
Please make sure that scripts "segm_mask_visual.py" is placed alongside with model, the saved checkpoint, and its configuration file. This means you should copy this script into the "detection" folder when conducting the visualization experiment. you need to use the following terminal command to start the visualization:
```
python segm_mask_visual.py --config <corresponding configuration file of models> --backbone <dsan-t or dsan-s or ...> --checkpoint <checkpoint file of the corresponding models> --palette <color map styles of dataset> --image_dir <folder containing source images> --out_dir <folder storing output visualization>
```

### Warnings
If these files do not work in the environment used for training models, please update the version of mmcv to 2.0.0 or higher, along with mmdet and mmsegmentation.
