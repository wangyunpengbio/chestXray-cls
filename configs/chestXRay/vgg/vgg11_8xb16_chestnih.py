_base_ = ['../chestnihDatasets224.py', '../../_base_/default_runtime.py']

# use different head for multilabel task
model = dict(
    type='ImageClassifier',
    backbone=dict(type='VGG', depth=11, norm_cfg=dict(type='BN'), num_classes=14),
    neck=None,
    head=dict(
        type='MultiLabelClsHead',
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
)

# load model pretrained on imagenet
load_from = 'https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_bn_batch256_imagenet_20210207-f244902c.pth'  # vgg11 bn

# optimizer
optimizer = dict(
    type='Adam',
    lr=1e-4,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(custom_keys={'.backbone.classifier': dict(lr_mult=10)}))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=40, gamma=0.5)
runner = dict(type='EpochBasedRunner', max_epochs=120)
