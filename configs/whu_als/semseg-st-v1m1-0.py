_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 8  # bs: total bs in all gpus
mix_prob = 0
empty_cache = False
enable_amp = True
find_unused_parameters = True
grid_size =0.3

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="ST-v1m1",
        downsample_scale=4,
        depths=[3, 3, 9, 3, 3],
        channels=[48, 96, 192, 384, 384],
        num_heads=[3, 6, 12, 24, 24],
        window_size=[0.1, 0.2, 0.4, 0.8, 1.6],
        up_k=3,
        grid_sizes=[0.02, 0.04, 0.08, 0.16, 0.32],
        quant_sizes=[0.005, 0.01, 0.02, 0.04, 0.08],
        rel_query=True,
        rel_key=True,
        rel_value=True,
        drop_path_rate=0.3,
        num_layers=5,
        concat_xyz=True,
        num_classes=7,
        ratio=0.25,
        k=16,
        prev_grid_size=0.3,
        sigma=1.0,
        stem_transformer=False,
        kp_ball_radius=0.02 * 2.5,
        kp_max_neighbor=34,
    ),
    criteria=[
        dict(
            type="CrossEntropyLoss",
            weight=[7.94949898,  1.80522474,  2.24312724,  2.86381332, 15.21661442,
                    1.85515968,  61.89627364],
            loss_weight=1.0,
            ignore_index=-1,
        )
    ],
)
# scheduler settings
epoch = 600
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(type="MultiStepLR", milestones=[0.6, 0.8], gamma=0.1)

# dataset settings
dataset_type = "WHUALSDataset"
data_root = "data/whu_als"

data = dict(
    num_classes=7,
    ignore_index=-1,
    names=[
    "Others", 
    "Ground",
    "Vegetation",
    "Lowveg",
    "Wire",
    "Building",
    # "Tree",
    "Light",
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            # dict(type="CenterShift", apply_z=True),
            # dict(
            #     type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            # ),
            # # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            # dict(type="RandomScale", scale=[0.9, 1.1]),
            # # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            # dict(type="RandomFlip", p=0.5),
            # # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            # dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "segment"),
                return_min_coord=True,
            ),
            # dict(type="SphereCrop", point_max=100000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            # dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("coord"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            # dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "segment"),
                return_min_coord=True,
            ),
            # dict(type="CenterShift", apply_z=False),
            # dict(type="NormalizeColor"),
            # dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("coord"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            # dict(type="CenterShift", apply_z=True),
            # dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord"),
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord"),
                ),
            ],
            aug_transform=[
                
            ],
        ),
    ),
)
