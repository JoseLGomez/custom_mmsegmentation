_base_ = [
    '../../_base_/models/daformer_sepaspp_mitb5.py',
    '../datasets/gta_to_cityscapes_512x512.py', '../../_base_/default_runtime.py',
    '../schedules/schedule_60k.py'
]


# runtime settingsMixVisionTransformer
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16000, metric='mIoU', pre_eval=True)

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

data = dict(samples_per_gpu=2, workers_per_gpu=4)

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='/data/new/Experiments/jlgomez/mmsegmentation/pretrained_weights/segformer/mit_b5_ready.pth'),
                ))
