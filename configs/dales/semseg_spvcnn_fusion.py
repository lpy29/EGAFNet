_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 24  # bs: total bs in all gpus
# batch_size_test = 1
mix_prob = 0
num_worker = 24  # total worker in all gpu
empty_cache = False
enable_amp = False
grid_size = 0.25
find_unused_parameters = True
clip_grad = 10.0
# seed = 46648935

# model settings
model = dict(
    type="DefaultFusionSegmentorV2",
    num_classes=8,
    backbone_out_channels=96,
    backbone=dict(
        type="SPVCNN",
        in_channels=3,
        out_channels=9,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 2, 2, 2, 2, 2, 2, 2),
    ),
    # img_backbone=dict(
    #     type="FPN",
    #     resnet = 'resnet18',
    #     layers=[2, 2, 2, 2],
    #     pretrained=True,
    #     replace_stride_with_dilation=[False, False, False],
    #     out_conv=False,
    #     out_channel=256,
    #     in_channels=[64, 128, 256 , -1],
    # ),
    img_backbone=dict(
        type="ResNet34",
    ),
    vit=dict(
        type="PVTv2",
        img_size=256,
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
    ),
    criteria=[
        dict(type="CrossEntropyLoss", weight = [
                1.43507485, 1.75723395, 11.89077808, 19.65859273, 21.47467105,
                15.96418609, 36.14609958, 2.50534948
            ],
             loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
        # dict(
        #     type="CrossEntropyLoss",
        #     # weight=[7.94949898,  1.80522474,  2.24312724,  2.86381332, 15.21661442,
        #     #         1.85515968,  61.89627364],
        #     loss_weight=1.0,
        #     ignore_index=-1,
        # )
    ],
)

# scheduler settings
epoch = 100
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
# optimizer = dict(type="Adam", lr=0.001, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# dataset settings
dataset_type = "DALESDataset"
data_root = "data/DALES/"
ignore_index = -1
names=[ 'Ground', 'Vegetation', 'Cars', 'Trucks', 'Power lines', 'Fences', 'Poles', 'Buildings']

data = dict(
    num_classes=8,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            # # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
            # # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
            # dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            # dict(type="RandomScale", scale=[0.9, 1.1]),
            # # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "segment","points_img","relative_z"),#,"control_point_img"),
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment","name","points_img","img","relative_z"),#,"control_point_img","control_img","control_feat"),
                feat_keys=("coord"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            # dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "segment","points_img","relative_z"),#,"control_point_img"),
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment","name","points_img","img","relative_z"),#,"control_point_img","control_img","control_feat"),
                feat_keys=("coord"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            # dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                keys=("coord", "segment","points_img","relative_z"),
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "segment","name","points_img","img","relative_z","index"),
                    feat_keys=("coord"),
                ),
            ],
            aug_transform=[
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[0],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         )
            #     ],
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[1 / 2],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         )
            #     ],
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[1],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         )
            #     ],
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[3 / 2],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         )
            #     ],
            ],
        ),
        ignore_index=ignore_index,
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="FusionSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=True),
]

