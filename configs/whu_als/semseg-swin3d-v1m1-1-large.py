_base_ = ["../_base_/default_runtime.py"]
# misc custom setting
batch_size = 8  # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = False
enable_amp = True
grid_size =0.3

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="Swin3D-v1m1",
        in_channels=3,
        num_classes=8,
        base_grid_size=0.02,
        depths=[2, 4, 9, 4, 4],
        channels=[80, 160, 320, 640, 640],
        num_heads=[10, 10, 20, 40, 40],
        window_sizes=[5, 7, 7, 7, 7],
        quant_size=4,
        drop_path_rate=0.3,
        up_k=3,
        num_layers=5,
        stem_transformer=True,
        down_stride=3,
        upsample="linear_attn",
        knn_down=True,
        cRSE="XYZ",
        fp16_mode=1,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", weight = [7.94949898,  1.80522474,  2.24312724,  2.86381332, 15.21661442,
        1.85515968,  4.03226445, 61.89627364],loss_weight=1.0, ignore_index=-1),
        # dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 3000
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="blocks", lr=0.0001)]

# dataset settings
dataset_type = "WHUALSDataset"
data_root = "data/whu_als"

data = dict(
    num_classes=13,
    ignore_index=-1,
    names=[
    "Others", 
    "Ground",
    "Vegetation",
    "Lowveg",
    "Wire",
    "Building",
    "Tree",
    "Light",
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            # dict(type="CenterShift", apply_z=True),
            # # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            # dict(type="RandomScale", scale=[0.8, 1.2]),
            # # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            # dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            # # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                keys=("coord",  "segment"),
                return_grid_coord=True,
                return_displacement=True,
            ),
            # dict(type="SphereCrop", point_max=80000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            # dict(type="NormalizeColor"),
            # # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord"),
                coord_feat_keys=("coord"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
        #     dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "segment"),
                return_grid_coord=True,
            ),
            # dict(type="CenterShift", apply_z=False),
            # dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord"),
                coord_feat_keys=("coord"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        # transform=[dict(type="CenterShift", apply_z=True), dict(type="NormalizeColor")],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord",  "segment"),
                return_grid_coord=True,
                return_displacement=True,
            ),
            crop=None,
            post_transform=[
                # dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord"),
                    coord_feat_keys=("coord"),
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
    ),
)
