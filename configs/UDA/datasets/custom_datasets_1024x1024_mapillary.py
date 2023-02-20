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
        img_scale=(1632, 1216),
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

custom_datasetA = dict(
    type='SynscapesDataset',
    img_dir='/datafast/118-1/Datasets/segmentation/new_synthetic_dataset/data_splits/Sysncapes_images_translated.txt',
    ann_dir='/datafast/118-1/Datasets/segmentation/new_synthetic_dataset/data_splits/Sysncapes_labels.txt',
    mode_txt=True,
    pipeline=train_pipeline)

custom_datasetB = dict(
    type='SynscapesDataset',
    img_dir='/datafast/118-1/Datasets/segmentation/GTA/gta5_rgb_translated_mapillary.txt',
    ann_dir='/datafast/118-1/Datasets/segmentation/GTA/gta5_gt_full.txt',
    mode_txt=True,
    pipeline=train_pipeline)

custom_datasetC = dict(
    type='SynscapesDataset',
    #data_root='data/synscapes/',
    img_dir='/datafast/118-1/Datasets/segmentation/Synscapes/img/rgb_translated_mapillary.txt',
    ann_dir='/datafast/118-1/Datasets/segmentation/Synscapes/img/labels_19.txt',
    mode_txt=True,
    pipeline=train_pipeline)

dataset_target = dict(
        type='CityscapesDataset',
        #data_root='data/cityscapes/',
        #img_dir='/datafast/118-1/Datasets/segmentation/cityscapes/leftimage8bit_val.txt',
        #ann_dir='/datafast/118-1/Datasets/segmentation/cityscapes/gtFine_val.txt',
        #img_dir='/datafast/118-1/Datasets/segmentation/bdd100k/val_bdd10k_images.txt',
        #ann_dir='/datafast/118-1/Datasets/segmentation/bdd100k/val_bdd10k_labels.txt',
        img_dir='/datafast/118-1/Datasets/segmentation/Mapillary_Vistas/v2.0/1617_val_images_aspect_1.33.txt',
        ann_dir='/datafast/118-1/Datasets/segmentation/Mapillary_Vistas/v2.0/1617_val_labels_aspect_1.33.txt',
        #ann_dir='/datafast/118-1/Datasets/segmentation/cityscapes/labels_multiple_classes/val/labels_vehicles.txt',
        #ann_dir='/datafast/118-1/Datasets/segmentation/cityscapes/labels_multiple_classes/val/labels_human_cycle.txt',
        #ann_dir='/datafast/118-1/Datasets/segmentation/cityscapes/labels_multiple_classes/val/labels_traffic.txt',
        #ann_dir='/datafast/118-1/Datasets/segmentation/cityscapes/labels_multiple_classes/val/labels_background.txt',
        mode_txt=True,
        pipeline=test_pipeline)


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=custom_datasetB,
    val=dataset_target,
    test=dataset_target)
