_base_ = [
    '../../_base_/models/deeplabv3plus_r50-d8.py',
    '../datasets/selftraining_gta+synscapes_to_cityscapes_1024x512_rcs.py', '../../_base_/default_runtime.py',
    '../schedules/schedule_60k.py', '../pseudolabeling/mpt_cot.py', '../pseudolabeling/uda.py'
]

norm_cfg = dict(type='BN')
model = dict(pretrained='/data/new/Experiments/jlgomez/mmsegmentation/baselines/gta+synscapes_deeplabV3plus_1024x512_60k_rcs/iter_60000.pth',
             backbone=dict(depth=101,
                           dilations=(1, 1, 1, 2),
                           strides=(1, 2, 2, 1),
                           multi_grid=(1, 2, 4)),
             decode_head=dict(num_classes=19,ignore_index=255, dilations=(1, 6, 12, 18),
                              sampler=dict(type='OHEMPixelSampler', min_kept=100000)),
             auxiliary_head=dict(num_classes=19))
runner = dict(type='IterBasedRunner', max_iters=16000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
