# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    #dict(type='Resize', img_scale=[(1024, 512),(1536, 768),(2048, 1024),(2560, 1280),(3072, 1536),(3584, 1792),(4096, 2048)], multiscale_mode='value'),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_ours = dict(
    type='SynscapesDataset',
    img_dir='/datafast/118-1/Datasets/segmentation/GTA/gta5_rgb_translated.txt',
    ann_dir='/datafast/118-1/Datasets/segmentation/GTA/gta5_gt_full.txt',
    mode_txt=True,
    pipeline=train_pipeline)

dataset_target = dict(
        type='CityscapesDataset',
        #data_root='data/cityscapes/',
        img_dir='/datafast/118-1/Datasets/segmentation/cityscapes/leftimage8bit_val.txt',
        ann_dir='/datafast/118-1/Datasets/segmentation/cityscapes/gtFine_val.txt',
        mode_txt=True,
        pipeline=test_pipeline)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dataset_ours,
    val=dataset_target,
    test=dataset_target)
