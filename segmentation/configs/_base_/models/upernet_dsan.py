# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='DSAN',
        embed_dims=[32, 64, 160, 256],
        kernel_sizes=[11, 11, 7, 5], 
        dw_kernel_sizes=[11,11,7,5],
        paddings=[5, 5, 3, 2], 
        strides=[1, 1, 1, 1],
        dilations=[1,1,1,1],
        groups=[1,4,8,8],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='UPerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
