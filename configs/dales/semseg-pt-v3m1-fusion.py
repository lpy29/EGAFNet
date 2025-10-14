_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 4  # bs: total bs in all gpus
num_worker = 8
mix_prob = 0.0
empty_cache = True
empty_cache_per_epoch=True
find_unused_parameters = True
enable_amp = False
grid_size =0.25
clip_grad = 1.0

# model settings
model = dict(
    type="DefaultFusionSegmentorV2",
    num_classes=8,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(128, 128, 128, 128, 128),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(128, 128, 128, 128),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        # pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
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
    criteria=[
        dict(type="CrossEntropyLoss", weight = [ 1,  1.51628214,  1.62871592, 10.72694891, 28.6257041 ,
       21.35381985, 13.66811657, 34.50614448,  2.46776615],loss_weight=1.0, ignore_index=-1),
        # dict(type="CrossEntropyLoss", weight = [7.94949898,  1.80522474,  2.24312724,  2.86381332, 15.21661442,
        # 1.85515968,  4.03226445, 61.89627364],loss_weight=1.0, ignore_index=-1),
        # dict(type="CrossEntropyLoss", weight = [7.94949898,  1.80522474,  2.24312724,  2.86381332, 15.21661442,1.85515968, 61.89627364],loss_weight=1.0, ignore_index=-1),
        # dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="CrossEntropyLoss",loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 60
eval_epoch = 60
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# dataset settings
# dataset settings
dataset_type = "DALESDataset"
data_root = "data/dales/"
ignore_index = -1
names = [
    'unknown',
    'Ground',
    'Vegetation',
    'Cars',
    'Trucks',
    'Power lines',
    'Fences',
    'Poles',
    'Buildings'
]



data = dict(
    num_classes=8,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            # dict(type="CenterShift", apply_z=True),
            # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
            # # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
            # dict(type="RandomScale", scale=[0.9, 1.1]),
            # # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            # dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                keys=("coord",  "segment","points_img"),
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment","name","points_img","img"),
                feat_keys=("coord"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
        loop=1,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            # dict(type="PointClip", point_cloud_range=(-51.2, -51.2, -4, 51.2, 51.2, 2.4)),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "segment"),
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment","name"),
                feat_keys=("coord", ),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
        loop=1,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "segment"),
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord"),
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord"),
                ),
            ],
            aug_transform=[
                # [dict(type="RandomScale", scale=[0.9, 0.9])],
                # [dict(type="RandomScale", scale=[0.95, 0.95])],
                # [dict(type="RandomScale", scale=[1, 1])],
                # [dict(type="RandomScale", scale=[1.05, 1.05])],
                # [dict(type="RandomScale", scale=[1.1, 1.1])],
                # [
                #     dict(type="RandomScale", scale=[0.9, 0.9]),
                #     dict(type="RandomFlip", p=1),
                # ],
                # [
                #     dict(type="RandomScale", scale=[0.95, 0.95]),
                #     dict(type="RandomFlip", p=1),
                # ],
                # [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                # [
                #     dict(type="RandomScale", scale=[1.05, 1.05]),
                #     dict(type="RandomFlip", p=1),
                # ],
                # [
                #     dict(type="RandomScale", scale=[1.1, 1.1]),
                #     dict(type="RandomFlip", p=1),
                # ],
            ],
        ),
        ignore_index=ignore_index,
        loop=1,
    ),
)

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="FusionSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
