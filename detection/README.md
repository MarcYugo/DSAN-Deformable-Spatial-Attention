### Trainig
Please use the script "dist_train.sh" to start model training. The sample is configured for a node with 8 GPUs for training the combination of mask-rcnn and dsan-t on COCO dataset. 
```bash
    #!/usr/bin/env bash
    
    CONFIG=configs/coco/mask_rcnn_dsan_t_fpn_1x.py
    GPUS=8
    PORT=${PORT:-29500}
    
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    torchrun --nproc_per_node=$GPUS --master_port=63667 \
        $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
```

### Eval

If you want to evaluate our models, please use the script "dist_test.sh". If you don't know how to do it, please check the user manual or documents of [mmdet-documents]([https://mmcv.readthedocs.io/en/v1.6.0/](https://mmdetection.readthedocs.io/en/v2.28.2/)).
