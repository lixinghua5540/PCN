# model settings
norm_cfg = dict(type='BN', requires_grad=True)

ckpt_path = "./pretrained/CVT_GeRSP_E10_GeRSP_200.pth"
model = dict(
    type='EncoderDecoder',
    backbone = dict(
        type='ResNetPretrain',
        ckpt_path=ckpt_path
        ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        channels=512,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide',stride=[128,128],crop_size=[256,256]))
