_base_ = ['../chestnihDatasets.py', '../../_base_/default_runtime.py']

# use different head for multilabel task
model = dict(
    type='ImageClassifier',
    backbone=dict(type='VGG', depth=16, num_classes=14),
    neck=None,
    # head=dict(
    #     type='MultiLabelLinearClsHead',
    #     num_classes=14,
    #     in_channels=2048,
    #     loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    head=dict(
        type='MultiLabelClsHead',
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
)

# load model pretrained on imagenet
load_from = 'https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_batch256_imagenet_20210208-db26f1a5.pth'  # noqa

# optimizer
optimizer = dict(
    type='Adam',
    lr=1e-5,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(custom_keys={'.backbone.classifier': dict(lr_mult=10)}))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=40, gamma=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=200)
