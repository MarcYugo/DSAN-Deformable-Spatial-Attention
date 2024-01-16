_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# optimizer
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
    neck=dict(in_channels=[32, 64, 160, 256]))
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05)
# batch_size = (8 GPUs) x (4 samples per GPU)
auto_scale_lr = dict(base_batch_size=4*8)

data = dict(samples_per_gpu=4)