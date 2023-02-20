_base_ = [
    '../../_base_/models/deeplabv3plus_r50-d8.py',
    '../datasets/gta_to_cityscapes_1024x512.py', '../../_base_/default_runtime.py',
    '../schedules/schedule_60k.py'
]

norm_cfg = dict(type='BN')
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101), decode_head=dict(num_classes=19,ignore_index=255),
              auxiliary_head=dict(num_classes=19))
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=90000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, metric='mIoU', pre_eval=True)