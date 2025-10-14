import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

import numpy as np
import os

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model
from tools.vis import write_ply
# import cv2
class LogScaling(nn.Module):
    def __init__(self):
        super(LogScaling, self).__init__()
        self.a = nn.Parameter(torch.tensor(10.0))  # 可学习参数 a
        self.b = nn.Parameter(torch.tensor(1.0))  # 可学习参数 b
        self.c = nn.Parameter(torch.tensor(1.0))  # 可学习参数 c

    def forward(self, x):
        log_part = self.a * torch.log1p(self.b * x)  # log1p = log(1 + x)
        linear_part = self.c * x
        return log_part #+ linear_part

@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)
        
@MODELS.register_module()
class DefaultFusionSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        img_backbone=None,
        vit=None,
        criteria=None,
    ):
        super().__init__()
        
          # 定义a为可学习参数
        self.a = nn.Parameter(torch.tensor(6.0))  # 初始化为1，你可以根据需要调整
        self.scaling_fn = LogScaling()
        
        self.seg_head =  nn.Sequential(
            nn.Linear(32*4, num_classes),
            # nn.ReLU(True),
            # nn.Linear(128, 9),
        )
        # self.seg_head = nn.Conv2d(32 * 4, num_classes, kernel_size=3, stride=1, padding=1)
        # 添加语义分割头（输出通道为7，表示8个类别）
        self.img_head = nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=1)
        
        self.backbone = build_model(backbone)
        self.img_backbone = build_model(img_backbone)
        self.vit = build_model(vit)
       


        # channels_3d = [64, 128, 256,512]
        channels_3d = [32,64, 128, 256]
        # channels_3d = [32,64, 128, 256]
        self.gl_fusion = nn.ModuleList()
        for i in range(4):
            self.gl_fusion.append(nn.Sequential(
                nn.Linear(128*2,32),
                nn.BatchNorm1d(32),
                nn.ReLU(True)
                ))
            
        self.z_fusion = nn.ModuleList()
        for i in range(4):
            self.z_fusion.append(nn.Sequential(
                nn.Linear(32*2,32),
                nn.BatchNorm1d(32),
                nn.ReLU(True)
                ))
        
        self.z_embed = nn.ModuleList()
        for i in range(4):
            self.z_embed.append(nn.Sequential(
                nn.Linear(1,32),
                nn.BatchNorm1d(32),
                nn.ReLU(True)
                ))
        
        self.zstd_embed = nn.ModuleList()
        for i in range(4):
            self.zstd_embed.append(nn.Sequential(
                nn.Linear(1,32),
                nn.BatchNorm1d(32),
                nn.ReLU(True)
                ))
            
        self.pointz_embed = nn.ModuleList()
        for i in range(4):
            self.pointz_embed.append(nn.Sequential(
                nn.Linear(1,32),
                nn.BatchNorm1d(32),
                nn.ReLU(True)
                ))
        
        self.point_mlp = nn.ModuleList()
        for i in range(4):
            self.point_mlp.append(nn.Sequential(
                nn.Linear(channels_3d[i],32),
                nn.BatchNorm1d(32),
                nn.ReLU(True)
                ))
            
        self.img_mlp = nn.ModuleList()
        for i in range(4):
            self.img_mlp.append(nn.Sequential(
                nn.Linear(128,32),
                nn.BatchNorm1d(32),
                nn.ReLU(True)
                ))
        
        self.control_mlp = nn.ModuleList()
        for i in range(4):
           self.control_mlp.append(nn.Sequential(
                nn.Linear(128,128),
                nn.BatchNorm1d(128),
                nn.Sigmoid()
                ))
            
        self.control_fusion = nn.ModuleList()
        for i in range(4):
            self.control_fusion.append(nn.Sequential(
                nn.Linear(128*2,1),
                nn.Sigmoid()
                ))
            
        self.z_adaptor = nn.ModuleList()
        for i in range(4):
            self.z_adaptor.append(nn.Sequential(
                nn.Linear(32*5,32*2),
                nn.BatchNorm1d(32*2),
                nn.ReLU(True),
                nn.Linear(32*2,1),
                nn.Sigmoid()
                ))

        
        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()
        for i in range(4):
            self.fcs1.append(nn.Sequential(nn.Linear(32 * 2, 32)))
            self.fcs2.append(nn.Sequential(nn.Linear(128, 1)))
        
        
        self.point_branch = nn.Linear(32,3)
        self.image_branch = nn.Linear(32,3)
        
        self.criteria = build_criteria(criteria)

    def forward(self,input_dict):

        img = input_dict["img"].to("cuda").permute(0,3,1,2)
        B,C,W,H =img.shape
        img_max = img[:,0,:,:].reshape(B,1,W,H)
        img_avg = img[:,2,:,:].reshape(B,1,W,H)
        img_std = img[:,3,:,:].reshape(B,1,W,H)
        

        # 归一化
        img_max = torch.relu(img_max)
        img_max= self.scaling_fn(img_max)
        input = torch.cat((img_max,img_avg, img_std), dim=1)
        
        indices = []
        dsm_zs = []
        dsm_stds = []
        control_feats = []
        batch_offset = input_dict["offset"]
        batch_size = len(batch_offset) 
        batch_offset = torch.cat((torch.tensor([0]).cuda(),batch_offset))  # 添加结束索引
        
        for i in range(batch_size):

            start_id = batch_offset[i]
            end_id = batch_offset[i + 1]
            indices.append(input_dict["points_img"][start_id:end_id])
            
            z_max_img = img[i][0]
            z_std_img = img[i][3]
            
            # get img_feat & dsm_features
            proj = input_dict["points_img"][start_id:end_id]
            ixs, iys = proj[:, 1].long(), proj[:, 0].long() # 这里需要确认索引的顺序???
            dsm_z = z_max_img[ixs,iys].float()
            dsm_zstd = z_std_img[ixs,iys].float()
            dsm_zs.append(dsm_z)
            dsm_stds.append(dsm_zstd)
            
            
        dsm_zs = torch.cat((dsm_zs),dim=0).reshape(-1,1)
        dsm_stds = torch.cat((dsm_stds),dim=0).reshape(-1,1)
        
        feat_2d_local, _ = self.img_backbone(input,indices)
        feat_2d_globel = self.vit(input,indices)
        feat_3d = self.backbone(input_dict)
        
        feat_all = []
        # feat_adaptor = None
        for i in range(4):
            # feat_2d_globel_transfer = self.vit_mlp[i](feat_2d_globel[i])
            feat_3d_transfer = self.point_mlp[i](feat_3d[i])
            feat_2d_transfer = self.gl_fusion[i](torch.cat([feat_2d_local[i],feat_2d_globel[i]],1))
            
    
            # # naive try : concat
            # feat_fuse = torch.cat([feat_3d_transfer,feat_2d_transfer],1)
            # feat_fuse = self.fcs1[i](feat_fuse)
            # naive try : add
            # feat_fuse = feat_3d_transfer+feat_2d_transfer
            # feat_2d_transfer = self.fc0[i](torch.cat([feat_2d[i],control_feats[:,:,3-i].permute(1,0)],1))
            # feat_3d_transfer = self.point_mlp[i](feat_3d[i])
            # feat_2d_transfer = self.img_mlp[i](feat_2d[i])

            
            # # gated
            z_embedding = self.z_embed[i](dsm_zs)
            zstd_embedding = self.zstd_embed[i](dsm_stds)
            pointz_embedding = self.pointz_embed[i](input_dict["relative_z"].reshape(-1,1))

            feat_adaptor = self.z_adaptor[i](torch.cat([feat_3d_transfer,feat_2d_transfer,z_embedding,zstd_embedding,pointz_embedding],1))
            feat_fuse = F.relu(feat_3d_transfer* feat_adaptor  + feat_2d_transfer * (1-feat_adaptor))

            feat_all.append(feat_fuse)
        


        seg_logits = self.seg_head(torch.cat(feat_all, 1))
        
        
        
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        # if isinstance(point, Point):
        #     feat = point.feat
        # else:
        #     feat = point
        # seg_logits = self.seg_head(feat)
        
        # train
        if self.training:
            # loss_pc = self.criteria.criteria[0](seg_logits, input_dict["segment"])
            # loss_pc_lovsz = self.criteria.criteria[2](seg_logits, input_dict["segment"])
            # # loss_pc = self.criteria(seg_logits, input_dict["segment"])
            # # #img_output
            # # loss_img = 0.0
            # # for j in range(len(image)):
            # #     W,H = image[j].shape
            # #     loss_img += self.criteria.criteria[1](img_output[j], torch.tensor(twod_label[j]).unsqueeze(0).long().cuda())
            # #     loss_img += self.criteria.criteria[2](img_output[j], torch.tensor(twod_label[j]).unsqueeze(0).long().cuda())
            # loss = loss_pc + loss_pc_lovsz#+ loss_img
            
            
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
