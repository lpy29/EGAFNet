import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

import numpy as np

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model

# import cv2

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
        
        # vit_checkpoint = torch.load("checkpoints/ckpt_pvt_t.pth", map_location='cpu')
        # self.vit.load_state_dict(vit_checkpoint['model'], strict=False)
        # checkpoint = torch.load("checkpoints/resnet34-333f7ec4.pth", map_location='cpu')
        # self.img_backbone.load_state_dict(checkpoint, strict=False)
        
        # # 使用 strict=False 加载参数，不严格对齐
        # state_dict = self.img_backbone.load_state_dict(checkpoint, strict=False)

        # # 打印加载的结果
        # print(f"Loaded parameters: {state_dict}")

        # # 获取哪些参数没有加载（missing_keys）
        # missing_keys = state_dict.missing_keys
        # unexpected_keys = state_dict.unexpected_keys

        # print(f"Missing keys: {missing_keys}")
        # print(f"Unexpected keys: {unexpected_keys}")

        # # 如果你想要查看具体哪些层的参数被加载了
        # print("\nLayers that have been loaded successfully:")
        # for name, param in self.img_backbone.load_state_dict.named_parameters():
        #     if name in checkpoint:
        #         print(f"Layer {name} has been loaded with parameters.")
        #     else:
        #         print(f"Layer {name} is missing or not loaded.")
       


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
        # vit_channel = [192,384,768,768]
        # self.vit_mlp = nn.ModuleList()
        # for i in range(4):
        #     self.vit_mlp.append(nn.Sequential(
        #         nn.Linear(vit_channel[i],128),
        #         nn.BatchNorm1d(128),
        #         nn.ReLU(True)
        #         ))
        
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
        
        
        self.criteria = build_criteria(criteria)

    def forward(self,input_dict):

        img = input_dict["img"].to("cuda").permute(0,3,1,2)
        B,C,W,H =img.shape
        img_max = img[:,0,:,:].reshape(B,1,W,H)
        img_avg = img[:,2,:,:].reshape(B,1,W,H)
        img_std = img[:,3,:,:].reshape(B,1,W,H)
        input = torch.cat((img_max,img_avg, img_std), dim=1)
        
        indices = []
        dsm_zs = []
        dsm_stds = []
        control_feats = []
        batch_offset = input_dict["offset"]
        batch_size = len(batch_offset) 
        batch_offset = torch.cat((torch.tensor([0]).cuda(),batch_offset))  # 添加结束索引
        
        for i in range(batch_size):
            # depth, dpt_feats = self.control.dpt_feature(img[i][0].detach().cpu().numpy())
            
            # dpt = cv2.imread('20241115-144904.jpg')
            # depth, dpt_feats = self.control.dpt_feature(dpt_fn = '', dpt = dpt)
            
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
        # control_feats = torch.cat((control_feats),dim=1)
        
        feat_2d_local, _ = self.img_backbone(input,indices)
        feat_2d_globel = self.vit(input,indices)
        # feat_2d_globel = self.img_vit(input,indices)
        # point = Point(input_dict)
        feat_3d = self.backbone(input_dict)
        # point = self.backbone(origins,layer_4_upsampled,layer_3_upsampled,layer_2_upsampled,layer_1_upsampled,proj,input_dict)
        
        feat_all = []
        # feat_adaptor = None
        for i in range(4):
            # feat_2d_globel_transfer = self.vit_mlp[i](feat_2d_globel[i])
            feat_3d_transfer = self.point_mlp[i](feat_3d[i])
            point_z = dsm_zs.reshape(-1)
            point_z = torch.clamp(point_z, min=1e-6) 
            point_z = self.a * torch.log(point_z + 1)
            z_embedding = self.z_embed[i](point_z.reshape(-1,1))
            
            feat_2d_gl = self.gl_fusion[i](torch.cat([feat_2d_local[i],feat_2d_globel[i]],1))
            
            feat_2d_transfer = self.z_fusion[i](torch.cat([feat_2d_gl,z_embedding],1))
            
            # feat_2d_transfer = self.img_mlp[i](feat_2d_local[i])
            
            # feat_2d_transfer = self.gl_fusion[i](torch.cat([feat_2d_local[i],feat_2d_globel[i]],1))
            
            # fusion_weight = self.control_fusion[i](torch.cat([feat_2d,control_feat],1))
            # feat_2d_transfer = fusion_weight *control_feat+(1-fusion_weight)*feat_2d_local[i]#self.img_mlp[i](torch.cat([feat_2d[i],control_feat],1))
            # feat_2d_transfer = self.img_mlp[i](feat_2d_transfer)
            # feat_2d_transfer = self.img_mlp[i](feat_2d_local[i])
            # # feat_2d_transfer = self.img_mlp[i](torch.cat([feat_2d_local[i],control_feats[:,:,3-i].permute(1,0)],1))
            # # naive try : concat
            feat_fuse = torch.cat([feat_3d_transfer,feat_2d_transfer],1)
            feat_fuse = self.fcs1[i](feat_fuse)
            # # naive try : add
            # feat_fuse = feat_3d_transfer+feat_2d_transfer
        #     # feat_2d_transfer = self.fc0[i](torch.cat([feat_2d[i],control_feats[:,:,3-i].permute(1,0)],1))
        #     feat_3d_transfer = self.point_mlp[i](feat_3d[i])
        #     feat_2d_transfer = self.img_mlp[i](feat_2d[i])
        #     # naive try : concat
            # feat_fuse = torch.cat([feat_3d_transfer,feat_2d_transfer],1)
            # feat_fuse = self.fcs1[i](feat_fuse)
            
            # # gated
            # z_embedding = self.z_embed[i](dsm_zs)
            # zstd_embedding = self.zstd_embed[i](dsm_stds)
            # pointz_embedding = self.pointz_embed[i](input_dict["relative_z"].reshape(-1,1))

            # feat_adaptor = self.z_adaptor[i](torch.cat([feat_3d_transfer,feat_2d_transfer,z_embedding,zstd_embedding,pointz_embedding],1))
            # # # # fuse_weight = torch.sigmoid(feat_adaptor)
            # # # # # fuse_weight = torch.sigmoid(self.fcs2[i](feat_fuse))
            
            # # # # # # fuse_weight = torch.sigmoid(self.fcs2[i](feat_adaptor))
            # # # # # # fuse_weight = feat_adaptor
            # # feat_fuse = torch.cat([feat_3d_transfer,feat_2d_transfer],1)
            # # # feat_fuse = self.fcs1[i](feat_fuse)
            # # # feat_fuse = F.relu(feat_fuse * feat_adaptor)
            # feat_fuse = F.relu(feat_3d_transfer* feat_adaptor  + feat_2d_transfer * (1-feat_adaptor))

            feat_all.append(feat_fuse)


        seg_logits = self.seg_head(torch.cat(feat_all, 1))
        # print(self.a)
        # seg_loss = self.seg_loss(seg_logits, label)
        
        
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
