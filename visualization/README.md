## Visualization
### CAM
Please make sure that scripts "cls_cam_visual.py" is placed alongside with model, the saved checkpoint, and its configuration file. This means you should copy this script into the "classification" folder when conducting the visualization experiment. Then, you need to use this terminal order to start the visualization.
```
python cls_cam_visual.py --model_name <dsan-t or dsan-s or ...> --images_dir <folder containing source images> --cams_dir <folder storing output visualization>
```

### 
