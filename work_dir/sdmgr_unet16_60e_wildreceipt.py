default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
randomness = dict(seed=None)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    visualization=dict(
        type='VisualizationHook',
        interval=1,
        enable=False,
        show=False,
        draw_gt=False,
        draw_pred=False))
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)
load_from = None
resume = False
visualizer = dict(
    type='KIELocalVisualizer', name='visualizer', is_openset=False)
wildreceipt_data_root = 'data/wildreceipt/'
wildreceipt_train = dict(
    type='WildReceiptDataset',
    data_root='data/wildreceipt/',
    metainfo='data/wildreceipt/class_list.txt',
    ann_file='train.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadKIEAnnotations'),
        dict(type='Resize', scale=(1024, 512), keep_ratio=True),
        dict(type='PackKIEInputs')
    ])
wildreceipt_test = dict(
    type='WildReceiptDataset',
    data_root='data/wildreceipt/',
    metainfo='data/wildreceipt/class_list.txt',
    ann_file='test.txt',
    test_mode=True,
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadKIEAnnotations'),
        dict(type='Resize', scale=(1024, 512), keep_ratio=True),
        dict(type='PackKIEInputs', meta_keys=('img_path', ))
    ])
optim_wrapper = dict(
    type='OptimWrapper', optimizer=dict(type='Adam', weight_decay=0.0001))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=60, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [dict(type='MultiStepLR', milestones=[40, 50], end=60)]
num_classes = 27
model = dict(
    type='SDMGR',
    kie_head=dict(
        type='SDMGRHead',
        visual_dim=16,
        num_classes=27,
        module_loss=dict(type='SDMGRModuleLoss'),
        postprocessor=dict(type='SDMGRPostProcessor')),
    dictionary=dict(
        type='Dictionary',
        dict_file=
        '/Users/lihardingnguyen/Developers/kie-ner/mmocr-kie-cv/configs/kie/sdmgr/../../../dicts/sdmgr_dict.txt',
        with_padding=True,
        with_unknown=True,
        unknown_token=None),
    backbone=dict(type='MobileViTUnet'),
    roi_extractor=dict(
        type='mmdet.SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=7),
        featmap_strides=[1]),
    data_preprocessor=dict(
        type='ImgDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadKIEAnnotations'),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    dict(type='PackKIEInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadKIEAnnotations'),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    dict(type='PackKIEInputs', meta_keys=('img_path', ))
]
val_evaluator = dict(
    type='F1Metric', mode='macro', num_classes=27, ignored_classes=[])
test_evaluator = dict(
    type='F1Metric', mode='macro', num_classes=27, ignored_classes=[])
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='WildReceiptDataset',
        data_root='data/wildreceipt/',
        metainfo='data/wildreceipt/class_list.txt',
        ann_file='train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadKIEAnnotations'),
            dict(type='Resize', scale=(1024, 512), keep_ratio=True),
            dict(type='PackKIEInputs')
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='WildReceiptDataset',
        data_root='data/wildreceipt/',
        metainfo='data/wildreceipt/class_list.txt',
        ann_file='test.txt',
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadKIEAnnotations'),
            dict(type='Resize', scale=(1024, 512), keep_ratio=True),
            dict(type='PackKIEInputs', meta_keys=('img_path', ))
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='WildReceiptDataset',
        data_root='data/wildreceipt/',
        metainfo='data/wildreceipt/class_list.txt',
        ann_file='test.txt',
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadKIEAnnotations'),
            dict(type='Resize', scale=(1024, 512), keep_ratio=True),
            dict(type='PackKIEInputs', meta_keys=('img_path', ))
        ]))
auto_scale_lr = dict(base_batch_size=4)
work_dir = './work_dir'
