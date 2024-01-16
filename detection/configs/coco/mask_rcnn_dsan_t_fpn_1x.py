_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='DSAN',
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        kernel_sizes=[11, 11, 7, 5], 
        dw_kernel_sizes=[11, 11, 7, 5],
        paddings=[5, 5, 3, 2], 
        strides=[1, 1, 1, 1],
        dilations=[1, 1, 1, 1],
        groups=[1, 4, 8, 8],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/dsan_t_encoder.pth')),
    neck=dict(
        type='FPN',
        in_channels=[32, 64, 160, 256],
        out_channels=256,
        num_outs=5))
# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05)
evaluation = dict(save_best='auto')
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=3,
    save_last=True,
)