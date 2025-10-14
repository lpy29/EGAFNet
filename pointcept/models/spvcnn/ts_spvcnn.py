"""
SPVCNN

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn

try:
    import torchsparse
    import torchsparse.nn as spnn
    import torchsparse.nn.functional as F
    # from torchsparse.nn.utils import get_kernel_offsets
    from torchsparse import PointTensor, SparseTensor
except ImportError:
    torchsparse = None


from pointcept.models.utils import offset2batch
from pointcept.models.builder import MODELS


def initial_voxelize(z):
    pc_hash = F.sphash(torch.floor(z.C).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = F.sphashquery(pc_hash, sparse_hash)
    counts = F.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = F.spvoxelize(torch.floor(z.C), idx_query, counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = F.spvoxelize(z.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features["idx_query"][1] = idx_query
    z.additional_features["counts"][1] = counts
    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if (
        z.additional_features is None
        or z.additional_features.get("idx_query") is None
        or z.additional_features["idx_query"].get(x.s) is None
    ):
        pc_hash = F.sphash(
            torch.cat(
                [
                    torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                    z.C[:, -1].int().view(-1, 1),
                ],
                1,
            )
        )
        sparse_hash = F.sphash(x.C)
        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features["idx_query"][x.s] = idx_query
        z.additional_features["counts"][x.s] = counts
    else:
        idx_query = z.additional_features["idx_query"][x.s]
        counts = z.additional_features["counts"][x.s]

    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if (
        z.idx_query is None
        or z.weights is None
        or z.idx_query.get(x.s) is None
        or z.weights.get(x.s) is None
    ):
        off = spnn.utils.get_kernel_offsets(2, x.s, 1, device=z.F.device)
        old_hash = F.sphash(
            torch.cat(
                [
                    torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                    z.C[:, -1].int().view(-1, 1),
                ],
                1,
            ),
            off,
        )
        pc_hash = F.sphash(x.C.to(z.F.device))
        idx_query = F.sphashquery(old_hash, pc_hash)
        weights = (
            F.calc_ti_weights(z.C, idx_query, scale=x.s[0]).transpose(0, 1).contiguous()
        )
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.0
            idx_query[:, 1:] = -1
        new_feat = F.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(
            new_feat, z.C, idx_query=z.idx_query, weights=z.weights
        )
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(
            new_feat, z.C, idx_query=z.idx_query, weights=z.weights
        )
        new_tensor.additional_features = z.additional_features

    return new_tensor


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, stride=stride, transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation, stride=1),
            spnn.BatchNorm(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


@MODELS.register_module()
class SPVCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 2, 2, 2, 2, 2, 2, 2),
    ):  # not implement
        super().__init__()

        assert (
            torchsparse is not None
        ), "Please follow `README.md` to install torchsparse.`"
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2

        self.stem = nn.Sequential(
            spnn.Conv3d(in_channels, base_channels, kernel_size=3, stride=1),
            spnn.BatchNorm(base_channels),
            spnn.ReLU(True),
            spnn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1),
            spnn.BatchNorm(base_channels),
            spnn.ReLU(True),
        )

        self.stage1 = nn.Sequential(
            *[
                BasicConvolutionBlock(
                    base_channels, base_channels, ks=2, stride=2, dilation=1
                ),
                ResidualBlock(base_channels, channels[0], ks=3, stride=1, dilation=1),
            ]
            + [
                ResidualBlock(channels[0], channels[0], ks=3, stride=1, dilation=1)
                for _ in range(layers[0] - 1)
            ]
        )

        self.stage2 = nn.Sequential(
            *[
                BasicConvolutionBlock(
                    channels[0], channels[0], ks=2, stride=2, dilation=1
                ),
                ResidualBlock(channels[0], channels[1], ks=3, stride=1, dilation=1),
            ]
            + [
                ResidualBlock(channels[1], channels[1], ks=3, stride=1, dilation=1)
                for _ in range(layers[1] - 1)
            ]
        )

        self.stage3 = nn.Sequential(
            *[
                BasicConvolutionBlock(
                    channels[1], channels[1], ks=2, stride=2, dilation=1
                ),
                ResidualBlock(channels[1], channels[2], ks=3, stride=1, dilation=1),
            ]
            + [
                ResidualBlock(channels[2], channels[2], ks=3, stride=1, dilation=1)
                for _ in range(layers[2] - 1)
            ]
        )

        self.stage4 = nn.Sequential(
            *[
                BasicConvolutionBlock(
                    channels[2], channels[2], ks=2, stride=2, dilation=1
                ),
                ResidualBlock(channels[2], channels[3], ks=3, stride=1, dilation=1),
            ]
            + [
                ResidualBlock(channels[3], channels[3], ks=3, stride=1, dilation=1)
                for _ in range(layers[3] - 1)
            ]
        )

        self.up1 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(channels[3], channels[4], ks=2, stride=2),
                nn.Sequential(
                    *[
                        ResidualBlock(
                            channels[4] + channels[2],
                            channels[4],
                            ks=3,
                            stride=1,
                            dilation=1,
                        )
                    ]
                    + [
                        ResidualBlock(
                            channels[4], channels[4], ks=3, stride=1, dilation=1
                        )
                        for _ in range(layers[4] - 1)
                    ]
                ),
            ]
        )

        self.up2 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(channels[4], channels[5], ks=2, stride=2),
                nn.Sequential(
                    *[
                        ResidualBlock(
                            channels[5] + channels[1],
                            channels[5],
                            ks=3,
                            stride=1,
                            dilation=1,
                        )
                    ]
                    + [
                        ResidualBlock(
                            channels[5], channels[5], ks=3, stride=1, dilation=1
                        )
                        for _ in range(layers[5] - 1)
                    ]
                ),
            ]
        )

        self.up3 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(channels[5], channels[6], ks=2, stride=2),
                nn.Sequential(
                    *[
                        ResidualBlock(
                            channels[6] + channels[0],
                            channels[6],
                            ks=3,
                            stride=1,
                            dilation=1,
                        )
                    ]
                    + [
                        ResidualBlock(
                            channels[6], channels[6], ks=3, stride=1, dilation=1
                        )
                        for _ in range(layers[6] - 1)
                    ]
                ),
            ]
        )

        self.up4 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(channels[6], channels[7], ks=2, stride=2),
                nn.Sequential(
                    *[
                        ResidualBlock(
                            channels[7] + base_channels,
                            channels[7],
                            ks=3,
                            stride=1,
                            dilation=1,
                        )
                    ]
                    + [
                        ResidualBlock(
                            channels[7], channels[7], ks=3, stride=1, dilation=1
                        )
                        for _ in range(layers[7] - 1)
                    ]
                ),
            ]
        )

        self.classifier = nn.Sequential(nn.Linear(channels[7], out_channels))

        self.point_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(base_channels, channels[3]),
                    nn.BatchNorm1d(channels[3]),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.Linear(channels[3], channels[5]),
                    nn.BatchNorm1d(channels[5]),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.Linear(channels[5], channels[7]),
                    nn.BatchNorm1d(channels[7]),
                    nn.ReLU(True),
                ),
            ]
        )
        
        self.point_transforms_1 = nn.Sequential(
                    nn.Linear(base_channels, channels[0]),
                    nn.BatchNorm1d(channels[0]),
                    nn.ReLU(True),
                )
        
        self.point_transforms_2 = nn.Sequential(
                    nn.Linear(base_channels, channels[1]),
                    nn.BatchNorm1d(channels[1]),
                    nn.ReLU(True),
                )
        
        self.point_transforms_3 = nn.Sequential(
                    nn.Linear(base_channels, channels[2]),
                    nn.BatchNorm1d(channels[2]),
                    nn.ReLU(True),
                )
        
        self.point_transforms_4 = nn.Sequential(
                    nn.Linear(base_channels, channels[3]),
                    nn.BatchNorm1d(channels[3]),
                    nn.ReLU(True),
                )
        
         # change the dsm feature to pc feature space
        self.fusion_mlp_dec0 = nn.Sequential(
                nn.Linear(64, channels[0]),
                nn.BatchNorm1d(channels[0]),
                nn.ReLU(True),
                )
        
        self.fusion_mlp_0 = nn.Sequential(
                nn.Linear(64, channels[0]),
                nn.BatchNorm1d(channels[0]),
                nn.ReLU(True),
                )
        
        # change z to another channels
        self.fusion_z_0 = nn.Sequential(
                nn.Linear(3, channels[0]),
                nn.BatchNorm1d(channels[0]),
                nn.ReLU(True),
                )
        # calculate the weight
        self.z_adaptor_0 = (
            nn.Sequential(
                nn.Linear(channels[0]*2, 1),
                nn.Sigmoid()
                )
        )
        
        self.fusion_mlp_dec1 = nn.Sequential(
                nn.Linear(64, channels[0]),
                nn.BatchNorm1d(channels[0]),
                nn.ReLU(True),
                )
        
        self.fusion_mlp_1 = nn.Sequential(
                nn.Linear(64, channels[0]),
                nn.BatchNorm1d(channels[0]),
                nn.ReLU(True),
                )
    
        self.fusion_z_1 = nn.Sequential(
                nn.Linear(3, channels[0]),
                nn.BatchNorm1d(channels[0]),
                nn.ReLU(True),
                )
        self.z_adaptor_1 = (
            nn.Sequential(
                nn.Linear(channels[0]*2, 1),
                nn.Sigmoid()
                )
        )
        
        self.fusion_mlp_dec2 = nn.Sequential(
                nn.Linear(128, channels[1]),
                nn.BatchNorm1d(channels[1]),
                nn.ReLU(True),
                )
        self.fusion_mlp_2 = nn.Sequential(
                nn.Linear(128, channels[1]),
                nn.BatchNorm1d(channels[1]),
                nn.ReLU(True),
                )

        self.fusion_z_2 = nn.Sequential(
                nn.Linear(3, channels[1]),
                nn.BatchNorm1d(channels[1]),
                nn.ReLU(True),
                )
        self.z_adaptor_2 = (
            nn.Sequential(
                nn.Linear(channels[1]*2, 1),
                nn.Sigmoid()
                )
        )
        
        self.fusion_mlp_dec3 = nn.Sequential(
                nn.Linear(256, channels[2]),
                nn.BatchNorm1d(channels[2]),
                nn.ReLU(True),
                )
        
        self.fusion_mlp_3 = nn.Sequential(
                nn.Linear(256, channels[2]),
                nn.BatchNorm1d(channels[2]),
                nn.ReLU(True),
                )

        self.fusion_z_3 = nn.Sequential(
                nn.Linear(3, channels[2]),
                nn.BatchNorm1d(channels[2]),
                nn.ReLU(True),
                )
        self.z_adaptor_3 = (
            nn.Sequential(
                nn.Linear(channels[2]*2, 1),
                nn.Sigmoid()
                )
        )

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def fuse_features(self, point, zstd ,image, image_feat, fusion_mlp_dec,fusion_mlp,fusion_z,z_adaptor):

        # 获取 batch 索引列
        batch_offset = torch.cumsum(point.C[:, -1].bincount(), dim=0).long()
        batch_size = len(batch_offset) 
        batch_offset = torch.cat((torch.tensor([0]).cuda(),batch_offset))  # 添加结束索引
        
        # 提前将图像和高度数据放入 GPU
        # image = [torch.tensor(img, device="cuda") for img in image]
        # zstd = [torch.tensor(z, device="cuda") for z in zstd]
        
        grid_coord = point.C
        point_feat = point.F
        
        dsm_feats = []
        dsm_zs = []
        dsm_stds = []
        point_zs = []
        
        for i in range(batch_size):
            start_id = batch_offset[i]
            end_id = batch_offset[i + 1]
            
            proj_ = grid_coord[start_id:end_id,:2]
            W = proj_[:,0].max() +1#- proj_[:,0].min() + 1  
            H = proj_[:,1].max() +1#- proj_[:,1].min() + 1
            img_feat = nn.functional.interpolate(image_feat[i], size=(W,H), mode='bilinear', align_corners=True).squeeze(0).permute(1,2,0)
            img = nn.functional.interpolate(torch.tensor(image[i]).cuda().unsqueeze(0).unsqueeze(0), size=(W,H), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
            std = nn.functional.interpolate(torch.tensor(zstd[i]).cuda().unsqueeze(0).unsqueeze(0), size=(W,H), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
            # img_feat = image_feat[i].squeeze(0).permute(1,2,0)
            # pc_feat = point.feat[start_id:end_id].clone()
            
            # get img_feat & dsm_features
            ixs, iys = proj_[:, 0].long(), proj_[:, 1].long()
            dsm_z = img[ixs,iys].float()
            dsm_zstd = std[ixs,iys].float()

            
            dsm_feat = img_feat[ixs,iys]
            # dsm_feat = dsm_feat / torch.norm(dsm_feat,dim=-1,keepdim=True)
            
            point_z = grid_coord[start_id:end_id,2].reshape(-1,1)
            point_zmin = torch.min(point_z)
            point_zmax = torch.max(point_z)
            z_norm = (point_z-point_zmin)/(point_zmax-point_zmin)
            z_norm = torch.clamp(z_norm, min=1e-8, max=0.99)  # 防止数值溢出

            dsm_zs.append(dsm_z)
            dsm_stds.append(dsm_zstd)
            dsm_feats.append(dsm_feat)
            point_zs.append(z_norm)
        
        dsm_feats = torch.cat((dsm_feats),dim=0)
        dsm_zs = torch.cat((dsm_zs),dim=0).reshape(-1,1)
        dsm_stds = torch.cat((dsm_stds),dim=0).reshape(-1,1)
        point_zs = torch.cat((point_zs),dim=0).reshape(-1,1)
        
        dsm_feats = fusion_mlp(dsm_feats)
        
        # change the channels to z related feat
        z = fusion_z(torch.cat((point_zs,dsm_zs,dsm_stds),dim=-1))
        # concat the z-related feat and point feat to get weight
        w = z_adaptor(torch.cat((z,point_feat),dim=-1))
        # weighted add 
        fused = nn.functional.relu(w *point_feat+(1-w)*dsm_feats)
        
        # fused_feats = torch.cat((dsm_feats,point_feat),dim=-1)
        # fused = fusion_mlp_dec(fused_feats)
        return fused



    def forward(self,data_dict):
        grid_coord = data_dict["grid_coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"]
        batch = offset2batch(offset)

        # x: SparseTensor z: PointTensor
        z = PointTensor(
            feat,
            torch.cat(
                [grid_coord.float(), batch.unsqueeze(-1).float()], dim=1
            ).contiguous(),
        )
        x0 = initial_voxelize(z)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        z1 = voxel_to_point(x4, z0)
        
        # f1 = voxel_to_point(x1, z0).F + self.point_transforms_1(z0.F)
        # f2 = voxel_to_point(x2, z0).F + self.point_transforms_2(z0.F)
        # f3 = voxel_to_point(x3, z0).F + self.point_transforms_3(z0.F)
        # f4 = voxel_to_point(x4, z0).F + self.point_transforms_4(z0.F)
        # return [f1,f2,f3,f4]
        return [voxel_to_point(x1, z0).F,voxel_to_point(x2, z0).F,voxel_to_point(x3, z0).F,voxel_to_point(x4, z0).F]
    
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.up1[0](y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)

        out = self.classifier(z3.F)
        return out
        # grid_coord = data_dict["grid_coord"]
        # feat = data_dict["feat"]
        # # proj = data_dict["proj"]
        # offset = data_dict["offset"]
        # batch = offset2batch(offset)
        
        # # layer_3_upsampled = [tensor.detach() for tensor in layer_3_upsampled]
        # # layer_2_upsampled = [tensor.detach() for tensor in layer_2_upsampled]
        # # layer_1_upsampled = [tensor.detach() for tensor in layer_1_upsampled]
        # # x: SparseTensor z: PointTensor
        # z = PointTensor(
        #     feat,
        #     torch.cat(
        #         [grid_coord.float(), batch.unsqueeze(-1).float()], dim=1
        #     ).contiguous(),
        # )
        # x0 = initial_voxelize(z)

        # x0 = self.stem(x0)
        # z0 = voxel_to_point(x0, z, nearest=False)
        # z0.F = z0.F

        # x1 = point_to_voxel(x0, z0)
        # # x1.F = self.fuse_features(x1, zstd ,image, img_embed, self.fusion_mlp_dec_0, self.fusion_mlp_0, self.fusion_z_0,self.z_adaptor_0)
        # x1 = self.stage1(x1)
        # x1.F = self.fuse_features(x1, zstd ,image, layer_1_upsampled, self.fusion_mlp_dec1, self.fusion_mlp_1, self.fusion_z_1,self.z_adaptor_1)
        # # p1 = voxel_to_point(x1, z0)
        # # p1.F = self.fuse_features(grid_coord,p1.F,offset,proj,zstd ,image, layer_1_upsampled, self.fusion_mlp_dec1, self.fusion_mlp_1, self.fusion_z_1,self.z_adaptor_1)
        # # p1_fused = point_to_voxel(x1, p1)
        
        # x2 = self.stage2(x1)
        # x2.F = self.fuse_features(x2, zstd ,image, layer_2_upsampled, self.fusion_mlp_dec2, self.fusion_mlp_2, self.fusion_z_2,self.z_adaptor_2)
        # x3 = self.stage3(x2)
        # x3.F = self.fuse_features(x3, zstd ,image, layer_3_upsampled, self.fusion_mlp_dec3, self.fusion_mlp_3, self.fusion_z_3,self.z_adaptor_3)
        # x4 = self.stage4(x3)
        
        # z1 = voxel_to_point(x4, z0)
        # z1.F = z1.F + self.point_transforms[0](z0.F)

        # y1 = point_to_voxel(x4, z1)
        # y1.F = self.dropout(y1.F)
        # y1 = self.up1[0](y1)
        # y1 = torchsparse.cat([y1, x3])
        # y1 = self.up1[1](y1)

        # y2 = self.up2[0](y1)
        # y2 = torchsparse.cat([y2, x2])
        # y2 = self.up2[1](y2)
        # z2 = voxel_to_point(y2, z1)
        # z2.F = z2.F + self.point_transforms[1](z1.F)

        # y3 = point_to_voxel(y2, z2)
        # y3.F = self.dropout(y3.F)
        # y3 = self.up3[0](y3)
        # y3 = torchsparse.cat([y3, x1])
        # y3 = self.up3[1](y3)

        # y4 = self.up4[0](y3)
        # y4 = torchsparse.cat([y4, x0])
        # y4 = self.up4[1](y4)
        # z3 = voxel_to_point(y4, z2)
        # z3.F = z3.F + self.point_transforms[2](z2.F)

        # out = z3.F#self.classifier(z3.F)
        # return out
