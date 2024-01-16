_base_ = [
    '../_base_/models/upernet_dsan.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
]
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/dsan_t_encoder.pth')),
    decode_head=dict(
        in_channels=[32, 64, 160, 256],
        num_classes=150
    ))

# optimizer
optimizer = dict(type='AdamW', lr=0.00006, weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=1e-6, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU')
data = dict(samples_per_gpu=4)
