_base_ = [
    '../../_base_/models/segformer_mit-b0.py',
    '../datasets/gta_to_cityscapes_1024x1024_single_txt.py', '../../_base_/default_runtime.py',
    '../schedules/schedule_60k_adam.py'
]


# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='/dataslow/storage/Experiments/jlgomez/mmsegmentation/pretrained_weights/segformer/mit_b5_ready.pth'),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512], ignore_index=19, num_classes=20),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024),
                  stride=(768, 768)))
