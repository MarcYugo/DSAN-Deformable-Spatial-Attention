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
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        kernel_sizes=[15, 13, 7, 5], 
        dw_kernel_sizes=[9, 7, 5, 5],
        paddings=[7, 6, 3, 2], 
        strides=[1, 1, 1, 1],
        dilations=[1, 1, 1, 1],
        groups=[4, 8, 16, 16],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[2, 2, 5, 3],
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/dsan_s_encoder.pth')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5))
# models are trained on 4 GPUs with 4 images per GPU
data = dict(samples_per_gpu=4)
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05)
evaluation = dict(save_best='auto')
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=3,
    save_last=True,
)