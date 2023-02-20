_base_ = [
    '../../_base_/models/deeplabv3plus_r50-d8.py',
    '../datasets/gta+synscapes_to_cityscapes_1024x512.py', '../../_base_/default_runtime.py',
    '../schedules/schedule_60k.py'
]

norm_cfg = dict(type='SyncBN')
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101), decode_head=dict(num_classes=19,ignore_index=255),
              auxiliary_head=dict(num_classes=19))
