# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (768, 768)
pseudolabels_scale=(2048, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
pseudolabel_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=pseudolabels_scale, ratio_range=(0.5, 2.0)),
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

dataset_A = dict(
    type='SynscapesDataset',
    img_dir='/datafast/118-1/Datasets/segmentation/new_synthetic_dataset/data_splits/gta5+Sysncapes+09-16-torstrasse+08-19-poblenou+08-19-poblenou_terrain_images_translated_filtered.txt',
    ann_dir='/datafast/118-1/Datasets/segmentation/new_synthetic_dataset/data_splits/gta5+Sysncapes+09-16-torstrasse+08-19-poblenou+08-19-poblenou_terrain_labels_filtered.txt',
    #ann_dir='/datafast/118-1/Datasets/segmentation/new_synthetic_dataset/data_splits/gta5+Sysncapes+09-16-torstrasse+08-19-poblenou+08-19-poblenou_terrain_labels_filtered_human_cycle.txt',
    #ann_dir='/datafast/118-1/Datasets/segmentation/new_synthetic_dataset/data_splits/gta5+Sysncapes+09-16-torstrasse+08-19-poblenou+08-19-poblenou_terrain_labels_filtered_traffic.txt',
    #ann_dir='/datafast/118-1/Datasets/segmentation/new_synthetic_dataset/data_splits/gta5+Sysncapes+09-16-torstrasse+08-19-poblenou+08-19-poblenou_terrain_labels_filtered_background.txt',
    mode_txt=True,
    pipeline=train_pipeline)

dataset_B = dict(
    type='SynscapesDataset',
    img_dir='/datafast/118-1/Datasets/segmentation/new_synthetic_dataset/data_splits/gta5+Sysncapes+09-16-torstrasse+08-19-poblenou+08-19-poblenou_terrain_images_translated_filtered.txt',
    ann_dir='/datafast/118-1/Datasets/segmentation/new_synthetic_dataset/data_splits/gta5+Sysncapes+09-16-torstrasse+08-19-poblenou+08-19-poblenou_terrain_labels_filtered_vehicles.txt',
    #ann_dir='/datafast/118-1/Datasets/segmentation/new_synthetic_dataset/data_splits/gta5+Sysncapes+09-16-torstrasse+08-19-poblenou+08-19-poblenou_terrain_labels_filtered_human_cycle.txt',
    #ann_dir='/datafast/118-1/Datasets/segmentation/new_synthetic_dataset/data_splits/gta5+Sysncapes+09-16-torstrasse+08-19-poblenou+08-19-poblenou_terrain_labels_filtered_traffic.txt',
    #ann_dir='/datafast/118-1/Datasets/segmentation/new_synthetic_dataset/data_splits/gta5+Sysncapes+09-16-torstrasse+08-19-poblenou+08-19-poblenou_terrain_labels_filtered_background.txt',
    mode_txt=True,
    pipeline=train_pipeline)

dataset_target = dict(
        type='CityscapesDataset',
        #data_root='data/cityscapes/',
        img_dir='/datafast/118-1/Datasets/segmentation/cityscapes/leftimage8bit_val.txt',
        ann_dir='/datafast/118-1/Datasets/segmentation/cityscapes/gtFine_val.txt',
        #ann_dir='/datafast/118-1/Datasets/segmentation/cityscapes/labels_multiple_classes/val/labels_vehicles.txt',
        #ann_dir='/datafast/118-1/Datasets/segmentation/cityscapes/labels_multiple_classes/val/labels_human_cycle.txt',
        #ann_dir='/datafast/118-1/Datasets/segmentation/cityscapes/labels_multiple_classes/val/labels_traffic.txt',
        #ann_dir='/datafast/118-1/Datasets/segmentation/cityscapes/labels_multiple_classes/val/labels_background.txt',
        mode_txt=True,
        pipeline=test_pipeline)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UdaDataset',
        source=dataset_A,
        pseudolabels=dict(
            type='CityscapesDataset',
            #data_root='',
            img_dir='',
            ann_dir='',
            split = '',
            mode_txt=True,
            img_suffix='',
            seg_map_suffix='',
            pipeline=pseudolabel_pipeline)),
    val=dataset_target,
    test=dataset_target)
