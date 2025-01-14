num_classes = 27

model = dict(
    type="SDMGR",
    kie_head=dict(
        type="SDMGRHead",
        visual_dim=16,
        num_classes=num_classes,
        module_loss=dict(type="SDMGRModuleLoss"),
        postprocessor=dict(type="SDMGRPostProcessor"),
    ),
    dictionary=dict(
        type="Dictionary",
        dict_file="{{ fileDirname }}/../../../dicts/sdmgr_dict.txt",
        with_padding=True,
        with_unknown=True,
        unknown_token=None,
    ),
)

train_pipeline = [
    dict(type="LoadKIEAnnotations"),
    dict(type="Resize", scale=(1024, 512), keep_ratio=True),
    dict(type="PackKIEInputs"),
]
test_pipeline = [
    dict(type="LoadKIEAnnotations"),
    dict(type="Resize", scale=(1024, 512), keep_ratio=True),
    dict(type="PackKIEInputs"),
]

val_evaluator = dict(
    type="F1Metric",
    mode="macro",
    num_classes=num_classes,
    ignored_classes=[],
)
test_evaluator = val_evaluator
