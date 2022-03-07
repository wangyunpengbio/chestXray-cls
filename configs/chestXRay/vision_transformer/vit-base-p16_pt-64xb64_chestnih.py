_base_ = ['../chestnihDatasets224.py', '../../_base_/default_runtime.py']

# use different head for multilabel task
# model = dict(
#     type='ImageClassifier',
#     backbone=dict(
#         type='ResNet',
#         depth=18,
#         num_stages=4,
#         out_indices=(3, ),
#         style='pytorch'),
#     neck=dict(type='GlobalAveragePooling'),
#     head=dict(
#         type='MultiLabelLinearClsHead',
#         num_classes=14,
#         in_channels=512,
#         loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
# )

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=14,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    )
)

# load model pretrained on imagenet
load_from = 'https://download.openmmlab.com/mmclassification/v0/vit/pretrain/vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth'  # vit 16 pretrain on 224

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
