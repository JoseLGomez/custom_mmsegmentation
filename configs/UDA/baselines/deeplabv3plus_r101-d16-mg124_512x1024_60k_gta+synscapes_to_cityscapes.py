_base_ = [
    '../../_base_/models/deeplabv3plus_r50-d8.py',
    '../datasets/gta+synscapes_to_cityscapes_1024x512.py', '../../_base_/default_runtime.py',
    '../schedules/schedule_60k.py'
]

norm_cfg = dict(type='BN')
model = dict(pretrained='/home/jlgomez/Repositories/mmsegmentation/R-103.pth',
             backbone=dict(depth=101,
                           dilations=(1, 1, 1, 2),
                           strides=(1, 2, 2, 1),
                           multi_grid=(1, 2, 4),
                           stem_channels=128,
                           norm_cfg=norm_cfg),
             decode_head=dict(num_classes=19,ignore_index=255, dilations=(1, 6, 12, 18),
                              norm_cfg=norm_cfg,
                              channels=256,
                              type='ASPPHead',
                              v3plus_bottleneck=True,
                              sampler=dict(type='OHEMPixelSampler', min_kept=100000)),
             auxiliary_head=dict(norm_cfg=norm_cfg,num_classes=19))
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=60000)
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=5000, metric='mIoU', pre_eval=True)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)