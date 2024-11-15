# DSA: Deformable Spatial Attention
<!--\[[Chinese version](https://gitcode.com/weixin_43385826/DSAN_Deformable_Spatial_Attention) \]  | [paper(preprint)]() -->
## Introduction
This is the implementation of DSA in PyTorch. DSA is a plug-and-play attention module, which combines deformable convolution and spatial attention. To improve the speed of deformable convolution with large kernel, we simplify the core operation of [DCNv3](https://github.com/OpenGVLab/InternImage/tree/master/classification/ops_dcnv3) by dropping modulation mask and changing bilinear interpolation on spatial domain into linear interpolation along the x or y axes, thereby reducing computation complex.

## Requirements
    CUDA>=11.6
    torch==1.13.0
    timm==0.4.12
    mmcv-full==1.6.0
    mmsegmentation==0.26.0
    mmdet==2.28.2
    yapf==0.40.0
    torchprofile==0.0.4

### DSCN
Install SDC before your experiments.

    cd classification/models/ops_dscn && python setup.py install

### Training

There are three vision tasks: classification, detection & instance segmentation, and semantic segmentation, corresponding to three folders listed from up to down. Each folder contains files and scripts used for training models. For detailed information about these scripts, please refer to the respective folders.

## Applications
You can downlad checkponints from following links.
### Image Classification

|Dataset|Model|Params(M)|FLOPs(G)|Top-1 Acc(%)|Download|
|-|-|-|-|-|-|
|IN-1K|DSAN-T|4.6|1.0|76.4|[ckpt](https://drive.google.com/file/d/13kzhzEBMaFJQ9bqSOyKx-uN0ufN45dkY/view?usp=sharing)|
||DSAN-T2|6.0|1.3|77.6|[ckpt](https://drive.google.com/file/d/1aiGgXTmHiorGoBXVQtuACGJhhYeVxrWl/view?usp=sharing)|
||DSAN-S|19.9|3.2|82.3|[ckpt](https://drive.google.com/file/d/1sBjIoUhwWZwmk8vnhkM8idu_faLRpliC/view?usp=sharing)|

### Semantic segmentation
|Dataset|Encoder|Decoder|Params(M)|FLOPs(G)|mIoU(MS)|Iterations|Download|
|-|-|-|-|-|-|-|-|
|ADE20K|DSAN-T|Hamburger|4.6|6.8|43.5|160K|[ckpt](https://drive.google.com/file/d/1SrWarGFufipUUbxCdkbXynz-kWaxGCUc/view?usp=sharing)|
||DSAN-S|-|20.7|23.0|48.8|160K|[ckpt](https://drive.google.com/file/d/1Is5_MYxWFIOUEG1AQ7wRMk4Xl8FDGRmT/view?usp=sharing)|
|Cityscapes|DSAN-T|-|4.6|52.6|80.0|80K|[ckpt](https://drive.google.com/file/d/1t8zbhSQ6FWbUWrV5pa8xc7aXU8s14wv8/view?usp=sharing)|
||DSAN-S|-|20.7|181.1|81.4|80K|[ckpt](https://drive.google.com/file/d/16O0cmW-sSpOumaJju99hagmd0wn3F9SK/view?usp=sharing)|

### Object Detection
|Dataset & Frame.|Backbone|Params(M)|GFLOPs(G)|mAP<sup>b</sup>(%)|mAP<sup>m</sup>(%)|Epochs|Download|
|-|-|-|-|-|-|-|-|
|COCO 2017|DSAN-T|24.3|188.6|42.6|38.9|12|[ckpt](https://drive.google.com/file/d/1tObK1d80nYuuI4Cq6VfXoPzsNXz0DVGq/view?usp=sharing)|
|Mask R-CNN|DSAN-S|39.5|235.4|46.1|41.5|12|[ckpt](https://drive.google.com/file/d/1G7EXX8iWhee51uEKYxa31DRD5UVG9c0o/view?usp=sharing)|

### License
Our implementation is based on [timm](https://github.com/huggingface/pytorch-image-models), [mmcv](https://github.com/open-mmlab/mmcv), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [mmdetection](https://github.com/open-mmlab/mmdetection) and [Hamburger](https://github.com/Gsunshine/Enjoy-Hamburger). The implementation of DSCN (core operation) mainly refers to [InternImage](https://github.com/OpenGVLab/InternImage). This repository is under [Apache-2.0 license](https://github.com/MarcYugo/DSAN_Deformable_Spatial_Attention?tab=Apache-2.0-1-ov-file).
