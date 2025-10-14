#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: basic_block.py
@time: 2021/12/16 20:34
'''
import torch
import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange



from torchvision.models.resnet import resnet34, resnet18

from pointcept.models.builder import MODELS

from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops import rearrange
import math

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FocusedLinearAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, linear=False,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))
        print('Linear Attention sr_ratio{} f{} kernel{}'.
              format(sr_ratio, focusing_factor, kernel_size))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
            else:
                kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        n = k.shape[1]

        k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q @ kv * z
        
        # """
        # attn_map = (x[] @ k.transpose(-2, -1))  # 注意力图，形状为 (B, num_heads, N, C // num_heads)
        # import matplotlib.pyplot as plt

        # # 选择一个批次和注意力头
        # batch_idx = 0
        # head_idx = 0

        # # 提取对应的注意力图
        # attn = attn_map[batch_idx, head_idx].cpu().detach().numpy()  # 形状为 (N, C // num_heads)

        # # 将注意力图还原为空间维度 (H, W)
        # H, W = int(N ** 0.5), int(N ** 0.5)  # 假设 N = H * W
        # attn = attn.reshape(H, W, -1)  # 形状为 (H, W, C // num_heads)

        # # 可视化某个通道的注意力图
        # channel_idx = 0
        # plt.imshow(attn[:, :, channel_idx], cmap='viridis')
        # plt.colorbar()
        # plt.title(f"Attention Map (Batch {batch_idx}, Head {head_idx}, Channel {channel_idx})")
        # plt.savefig('test.png', bbox_inches='tight', dpi=300)  # 保存图像
        # """
        # import matplotlib.pyplot as plt
        # # 取 batch 中的第一个样本
        # kv_vis = kv[0].detach().cpu().numpy()  # (num_heads, N, C)

        # # 将 kv 重塑为 (num_heads, H, W, C)
        # kv_vis = kv_vis.reshape( H, W, -1)

        # # 可视化每个注意力头的 kv
        # plt.figure(figsize=(15, 5 * self.num_heads))
        # for i in range(self.num_heads):
        #     plt.subplot(self.num_heads, 1, i + 1)
        #     plt.imshow(kv_vis, cmap='viridis')  # 取通道均值
        #     plt.title(f"Attention Map (Head {i + 1})")
        #     plt.colorbar()

        # plt.tight_layout()
        # plt.savefig('test.png', bbox_inches='tight', dpi=300)  # 保存图像
        # plt.close()  # 关闭图像，避免内存泄漏

        if self.sr_ratio > 1 or self.linear:
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=N, mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)
        x = x.transpose(1, 2).reshape(B, N, C)
        v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
        x = x + self.dwc(v).reshape(B, C, N).permute(0, 2, 1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_patches, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False,
                 focusing_factor=3, kernel_size=5, attn_type='L'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        assert attn_type in ['L', 'S']
        if attn_type == 'L':
            self.attn = FocusedLinearAttention(
                dim, num_patches,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear,
                focusing_factor=focusing_factor, kernel_size=kernel_size)
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

@MODELS.register_module("PVTv2")
class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], la_sr_ratios='8421', num_stages=4, linear=False,
                 focusing_factor=3, kernel_size=5, attn_type='LLLL'):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        attn_type = 'LLLL' if attn_type is None else attn_type
        
        self.fc = nn.ModuleList()
        for i in range(num_stages):
            self.fc.append(nn.Conv2d(embed_dims[i],128,kernel_size=3, stride=1, padding=1, bias=False))
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_patches=patch_embed.num_patches, num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i] if attn_type[i] == 'S' else int(la_sr_ratios[i]), linear=linear,
                focusing_factor=focusing_factor, kernel_size=kernel_size, attn_type=attn_type[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        feats = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            # if i != self.num_stages - 1:
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            feats.append(x)
        return feats
        return x.mean(dim=1)

    def forward(self, x, indices):
        B,_,W,H = x.shape
        x = self.forward_features(x)
        # x = self.head(x)
        
        temp = [[] for k in range(len(x))]
       
        for i in range(B):
            for k in range(len(x)):
                layer = F.interpolate(self.fc[k](x[k]), size=(W,H), mode='bilinear', align_corners=True)
                ss = layer.permute(0, 2, 3, 1).contiguous()
                temp[k].append(ss[i][indices[i][:, 1], indices[i][:, 0]].contiguous())

        for k in range(len(x)):
            x[k] = torch.cat(temp[k], 0).contiguous()

        return x

# # helpers
# def pair(t):
#     return t if isinstance(t, tuple) else (t, t)

# # classes
# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn
#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)

# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#     def forward(self, x):
#         return self.net(x)

# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim = -1)
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x):
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)

# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
#             ]))
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x

# class VitSegNet(nn.Module):
#     def __init__(self,
#                 image_size=144,
#                 patch_h_size=8,
#                 patch_w_size=8,
#                 channels=64,
#                 dim=512,
#                 depth=5,
#                 heads=16,
#                 output_channels=1024,
#                 expansion_factor=4,
#                 dim_head=64,
#                 dropout=0.,
#                 emb_dropout=0.,
#                 is_with_shared_mlp=False):  # mlp_dim is corresponding to expansion factor
#         super().__init__()
#         image_height, image_width = pair(image_size)
#         self.image_size = image_size
#         patch_height, patch_width = pair((patch_h_size, patch_w_size))

#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

#         num_patches = (image_height // patch_height) * (image_width // patch_width)  #
#         patch_dim = channels * patch_height * patch_width
#         temp_h = int(image_size / patch_h_size)
#         temp_w = int(image_size / patch_w_size)
#         # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

#         # h: patch number in height axis; w: patch number in width axis
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
#             nn.Linear(patch_dim, dim),
#         )

#         # Without cls token
#         # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         # self.dropout = nn.Dropout(emb_dropout)

#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         mlp_dim = int(dim*expansion_factor)

#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

#         self.rearrange = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = temp_h, w=temp_w, p1 = patch_h_size, p2 = patch_w_size)
        
#         out_in_channels = int(dim/(patch_h_size*patch_w_size))

        
#         if is_with_shared_mlp:
#             self.is_with_shared_mlp = True
#             self.shared_mlp = nn.Conv2d(in_channels=out_in_channels, out_channels=output_channels, kernel_size=1)
#         else:
#             self.is_with_shared_mlp = False

        
#     def forward(self, img):
#         # img = F.interpolate(img, size=(self.image_size, self.image_size), mode='bilinear', align_corners=True)
#         x = self.to_patch_embedding(img)
#         _, n, _ = x.shape  # torch.Size([4, 324, 512])  [batch_size, patch_number, patch_channels]

#         # Without cls token
#         # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
#         # x = torch.cat((cls_tokens, x), dim=1)
#         # x += self.pos_embedding[:, :(n + 1)]
        
#         x += self.pos_embedding[:, :n]  # we embedded [patch_number] positions, then concat it to the image features
        
#         x = self.dropout(x)
#         x = self.transformer(x)   # x.shape = torch.Size([4, 324, 512]) # after transformer
        
#         x = self.rearrange(x)     # x.shape = torch.Size([4, 8, 144, 144])  # after rearrange

#         if self.is_with_shared_mlp:
#             x = self.shared_mlp(x)
#             # print(f'4: {x.shape}')  # torch.Size([4, 8, 144, 144])
       
#         return x

# class SparseBasicBlock(spconv.SparseModule):
#     def __init__(self, in_channels, out_channels, indice_key):
#         super(SparseBasicBlock, self).__init__()
#         self.layers_in = spconv.SparseSequential(
#             spconv.SubMConv3d(in_channels, out_channels, 1, indice_key=indice_key, bias=False),
#             nn.BatchNorm1d(out_channels),
#         )
#         self.layers = spconv.SparseSequential(
#             spconv.SubMConv3d(in_channels, out_channels, 3, indice_key=indice_key, bias=False),
#             nn.BatchNorm1d(out_channels),
#             nn.LeakyReLU(0.1),
#             spconv.SubMConv3d(out_channels, out_channels, 3, indice_key=indice_key, bias=False),
#             nn.BatchNorm1d(out_channels),
#         )

#     def forward(self, x):
#         identity = self.layers_in(x)
#         output = self.layers(x)
#         return output.replace_feature(F.leaky_relu(output.features + identity.features, 0.1))
    

# def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
#                  relu=True):
#     assert (kernel % 2) == 1, \
#         'only odd kernel is supported but kernel = {}'.format(kernel)

#     layers = []
#     layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
#                             bias=not bn))
#     if bn:
#         layers.append(nn.BatchNorm2d(ch_out))
#     if relu:
#         layers.append(nn.LeakyReLU(0.2, inplace=True))

#     layers = nn.Sequential(*layers)

#     return layers


# def convt_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0,
#                   bn=True, relu=True):
#     assert (kernel % 2) == 1, \
#         'only odd kernel is supported but kernel = {}'.format(kernel)

#     layers = []
#     layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding,
#                                      output_padding, bias=not bn))
#     if bn:
#         layers.append(nn.BatchNorm2d(ch_out))
#     if relu:
#         layers.append(nn.LeakyReLU(0.2, inplace=True))

#     layers = nn.Sequential(*layers)

#     return layers

# @MODELS.register_module("ResNet18")
# class ResNet18(nn.Module):
#     def __init__(self):
#         super(ResNet18, self).__init__()
#         net = resnet18(pretrained=False)

#         # 修改第一层卷积层，使其接受单通道输入
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#         )


#         self.bn1 = net.bn1
#         self.relu = net.relu
#         # self.maxpool = net.maxpool
#         self.layer1 = net.layer1
#         self.layer2 = net.layer2
#         self.layer3 = net.layer3
#         self.layer4 = net.layer4
        
#         # Top layer
#         self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        
#         self.latlayer1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
#         self.latlayer2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
#         self.latlayer3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        
#         self.feature_layer = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        
#         self.gn = nn.GroupNorm(256, 256)
        
#         # self.gn12 = nn.GroupNorm(self.inplanes * self.expansion, self.inplanes * self.expansion)
#         # self.gn21 = nn.GroupNorm(256, 256)
#         # self.gn22 = nn.GroupNorm(self.inplanes * self.expansion, self.inplanes * self.expansion)
        
#         self.dec4 = convt_bn_relu(512, 256, kernel=3, stride=2,
#                                   padding=1, output_padding=1)
#         # 1/4
#         self.dec3 = convt_bn_relu(256+256, 128, kernel=3, stride=2,
#                                   padding=1, output_padding=1)
#         # 1/2
#         self.dec2 = convt_bn_relu(128+128, 64, kernel=3, stride=2,
#                                   padding=1, output_padding=1)
#         # self.dec2 = convt_bn_relu(128, 64, kernel=3, stride=2,
#         #                           padding=1, output_padding=1)

#         # 1/1
#         self.dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1,
#                                   padding=1)
        
        
#         #通道数调整
#         # self.conv_fusion = nn.Conv2d(64+512, 128, kernel_size=1, stride=1, padding=0)
#         # self.bn_fusion = nn.BatchNorm2d(128)
#         # self.relu_fusion = nn.ReLU(inplace=True)

        
#     def _concat(self, fd, fe, dim=1):
#         # Decoder feature may have additional padding
#         _, _, Hd, Wd = fd.shape
#         _, _, He, We = fe.shape

#         # Remove additional padding
#         if Hd > He:
#             h = Hd - He
#             fd = fd[:, :, :-h, :]

#         if Wd > We:
#             w = Wd - We
#             fd = fd[:, :, :, :-w]

#         f = torch.cat((fd, fe), dim=dim)

#         return f

#     def _upsample(self, x, h, w):
#         return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

#     def _upsample_add(self, x, y):
#         '''Upsample and add two feature maps.
#         Args:
#           x: (Variable) top feature map to be upsampled.
#           y: (Variable) lateral feature map.
#         Returns:
#           (Variable) added feature map.
#         Note in PyTorch, when input size is odd, the upsampled feature map
#         with `F.upsample(..., scale_factor=2, mode='nearest')`
#         maybe not equal to the lateral feature map size.
#         e.g.
#         original input size: [N,_,15,15] ->
#         conv2d feature map size: [N,_,8,8] ->
#         upsampled feature map size: [N,_,16,16]
#         So we choose bilinear upsample which supports arbitrary output sizes.
#         '''
#         _, _, H, W = y.size()
#         return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    
#     def forward(self, x):
        
#         _,_,W,H = x.shape
        
#         x = self.conv1(x)
#         # origin = x# 81 * 81
#         x = self.bn1(x)
#         x = self.relu(x)
#         origin = x
#         # x = self.maxpool(x)
        
#         layer1_out = self.layer1(x) # 81 * 81
#         layer2_out = self.layer2(layer1_out) # 40 * 40
#         layer3_out = self.layer3(layer2_out) # 20 * 20
#         layer4_out = self.layer4(layer3_out) # 10 * 10
        
#         dec_4_upsampled = F.interpolate(layer4_out, size=(W,H), mode='bilinear', align_corners=True)
#         dec_3_upsampled = F.interpolate(layer3_out, size=(W,H), mode='bilinear', align_corners=True)
#         dec_2_upsampled = F.interpolate(layer2_out, size=(W,H), mode='bilinear', align_corners=True)
#         # dec_2 = self.dec2(layer2_out) # 80 * 80
#         # dec_1 = self.dec1(self._concat(dec_2, layer1_out)) # 80 * 80
        
#         # p4 = self.toplayer(layer4_out)
#         # p3 = self._upsample_add(p4, self.latlayer1(layer3_out))
#         # p2 = self._upsample_add(p3, self.latlayer2(layer2_out))
#         # p1 = self._upsample_add(p2, self.latlayer3(layer1_out))
        
#         # s4 = self._upsample(F.relu(self.gn(p4)), W,H )
#         # s3 = self._upsample(F.relu(self.gn(p3)), W,H )
#         # s2 = self._upsample(F.relu(self.gn(p2)), W,H )
#         # s1 = F.relu(self.gn(p1))
        
#         # fea_upsample = self.feature_layer(s2 + s3 + s4 + s1)
        
#         dec_4 = self.dec4(layer4_out) # 20 * 20
#         dec_3 = self.dec3(self._concat(dec_4, layer3_out)) # 40 * 40
#         dec_2 = self.dec2(self._concat(dec_3, layer2_out)) # 80 * 80
#         dec_1 = self.dec1(self._concat(dec_2, layer1_out)) # 80 * 80
        
#         # layer4 = F.interpolate(layer4_out, size=(W,H), mode='bilinear', align_corners=True)
#         # dec_4_upsampled = F.interpolate(dec_4, size=(W,H), mode='bilinear', align_corners=True)
#         # dec_3_upsampled = F.interpolate(dec_3, size=(W,H), mode='bilinear', align_corners=True)
#         # dec_2_upsampled = F.interpolate(dec_2, size=(W,H), mode='bilinear', align_corners=True)
#         # dec_2 = self.dec2(layer2_out) # 80 * 80
#         # dec_1 = self.dec1(self._concat(dec_2, layer1_out)) # 80 * 80
        
#         # print(dec_4.shape, dec_3.shape, dec_2.shape, dec_1.shape)
#         # exit()
        
#         # return origin,x1_upsampled,x2_upsampled, x3_upsampled,x4_upsampled, dec_1 
#         # return origin, x4_upsampled, x3_upsampled,x2_upsampled, x1_upsampled,dec_1 
#         return origin,None, layer4_out,layer3_out,layer2_out, layer1_out, dec_1
#         # return origin,None, dec_4_upsampled,dec_3_upsampled,dec_2_upsampled, layer1_out, dec_1
    
# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)


# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError(
#                 'BasicBlock only supports groups=1 and base_width=64')
#         # if dilation > 1:
#         #     raise NotImplementedError(
#         #         "Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         # self.relu = nn.GELU()
#         self.conv2 = conv3x3(planes, planes, dilation=dilation)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out

# @MODELS.register_module("FPN")
# class FPNWrapper(nn.Module):
#     def __init__(self,
#                 resnet = 'resnet18',
#                 block = BasicBlock,
#                 layers=[2, 2, 2, 2],
#                 pretrained=True,
#                 replace_stride_with_dilation=[False, False, False],
#                 out_conv=False,
#                 out_channel=128,
#                 in_channels=[64, 128, 256, 512],
#                 cfg=None,
#                 norm_layer=None):
#         super(FPNWrapper, self).__init__()
#         # net = resnet34(pretrained=True)
        
#         self.cfg = cfg
#         self.in_channels = in_channels

#         # ResNet Details
#         self.groups = 1
#         self.base_width = 64
#         self.inplanes = 64
#         self.dilation = 1
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer

#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

#         # Layer-0
#         # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
#         # 修改第一层卷积层，使其接受单通道输入
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#         )
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

#         self.layer1 = self._make_layer(block, in_channels[0], layers[0], stride=1)
#         self.layer2 = self._make_layer(block, in_channels[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0])
#         if in_channels[2] > 0:
#             self.layer3 = self._make_layer(block, in_channels[2], layers[2], stride=2,
#                                         dilate=replace_stride_with_dilation[1])
#         if in_channels[3] > 0:
#             self.layer4 = self._make_layer(block, in_channels[3], layers[3], stride=2,
#                                            dilate=replace_stride_with_dilation[2])
        
#         # self.vit1 = VitSegNet(image_size=64,
#         #                       patch_h_size=1,
#         #                       patch_w_size=1,
#         #                       channels=64,
#         #                       dim=64,)
#         # # self.output_conv1 = nn.Conv2d(in_channels=8, out_channels=64, kernel_size=1)
#         # self.fusion_mlp1 = nn.Sequential(
#         #                     nn.Conv2d(128, 64, kernel_size=1),
#         #                     nn.BatchNorm2d(64),
#         #                     nn.ReLU(True),
#         #                     )
        
#         # self.vit2 = VitSegNet(image_size=41,
#         #                       patch_h_size=1,
#         #                       patch_w_size=1,
#         #                       channels=128,
#         #                       dim=128,)
#         # self.output_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
#         # self.fusion_mlp2 = nn.Sequential(
#         #                     nn.Conv2d(256, 128, kernel_size=1),
#         #                     nn.BatchNorm2d(128),
#         #                     nn.ReLU(True),
#         #                     )
        
#         # self.vit3 = VitSegNet(image_size=21,
#         #                       patch_h_size=1,
#         #                       patch_w_size=1,
#         #                       channels=256,
#         #                       dim=256,)
#         # self.output_conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1)
#         # self.fusion_mlp3 = nn.Sequential(
#         #                     nn.Conv2d(512, 256, kernel_size=1),
#         #                     # nn.Linear(512, 256),
#         #                     nn.BatchNorm2d(256),
#         #                     nn.ReLU(True),
#         #                     )
#         # self.layer1 = net.layer1
#         # self.layer2 = net.layer2
#         # if in_channels[2] > 0:
#         #     self.layer3 = net.layer3
#         # if in_channels[3] > 0:
#         #     self.layer4 = net.layer4
        
#         self.expansion = block.expansion
#         #!!!IMPORTANT!!! store output feature for Vision transformer
#         self.out = None
#         if out_conv:
#             out_channel = 512
#             for chan in reversed(self.in_channels):
#                 if chan < 0: continue
#                 out_channel = chan
#                 break
#             self.out = conv1x1(
#                 out_channel * self.expansion, cfg.featuremap_out_channel)

#         # upsample and connect
#         # Top layer
#         self.toplayer = nn.Conv2d(out_channel * self.expansion, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

#         # Smooth layers
#         self.smooth1 = nn.Conv2d(self.inplanes * self.expansion, self.inplanes * self.expansion, kernel_size=3, stride=1, padding=1)
#         self.smooth2 = nn.Conv2d(self.inplanes * self.expansion, self.inplanes * self.expansion, kernel_size=3, stride=1, padding=1)
#         self.smooth3 = nn.Conv2d(self.inplanes * self.expansion, self.inplanes * self.expansion, kernel_size=3, stride=1, padding=1)

#         # Lateral layers
#         if in_channels[3] > 0:
#             self.latlayer1 = nn.Conv2d(self.in_channels[2] * self.expansion, self.inplanes * self.expansion, kernel_size=1, stride=1, padding=0)
#             self.latlayer2 = nn.Conv2d(self.in_channels[1] * self.expansion, self.inplanes * self.expansion, kernel_size=1, stride=1, padding=0)
#             self.latlayer3 = nn.Conv2d(self.in_channels[0] * self.expansion, self.inplanes * self.expansion, kernel_size=1, stride=1, padding=0)
#         elif in_channels[2] > 0:
#             self.latlayer1 = nn.Conv2d(self.in_channels[1] * self.expansion, self.inplanes * self.expansion, kernel_size=1, stride=1, padding=0)
#             self.latlayer2 = nn.Conv2d(self.in_channels[0] * self.expansion, self.inplanes * self.expansion, kernel_size=1, stride=1, padding=0)
#         else:
#             self.latlayer1 = nn.Conv2d(self.in_channels[0] * self.expansion, self.inplanes * self.expansion, kernel_size=1, stride=1, padding=0)

#         self.semantic_branch = nn.Conv2d(self.inplanes * self.expansion, int(self.inplanes * self.expansion * 0.5), kernel_size=3, stride=1, padding=1)
#         self.semantic_branch2 = nn.Conv2d(self.inplanes * self.expansion, int(self.inplanes * self.expansion * 0.5),
#                                          kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(self.inplanes * self.expansion, self.inplanes * self.expansion, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(self.inplanes * self.expansion, self.inplanes * self.expansion, kernel_size=3, stride=1, padding=1)

#         self.feature_layer = nn.Conv2d(int(self.inplanes * self.expansion * 0.5), 128, kernel_size=1, stride=1, padding=0)
#         self.feature_bn = nn.BatchNorm2d(128)
#         self.feature_gelu = nn.GELU()
#         self.output_layer_binary_seg = nn.Conv2d(8, 3, kernel_size=1, stride=1, padding=0)
#         self.output_layer_endp = nn.Conv2d(int(self.inplanes * self.expansion * 0.5), 1, kernel_size=1, stride=1, padding=0)
#         self.gn11 = nn.GroupNorm(int(self.inplanes * self.expansion * 0.5), int(self.inplanes * self.expansion * 0.5))
#         self.gn12 = nn.GroupNorm(self.inplanes * self.expansion, self.inplanes * self.expansion)
#         self.gn21 = nn.GroupNorm(int(self.inplanes * self.expansion * 0.5), int(self.inplanes * self.expansion * 0.5))
#         self.gn22 = nn.GroupNorm(self.inplanes * self.expansion, self.inplanes * self.expansion)

#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer))

#         return nn.Sequential(*layers)

#     def _upsample(self, x, h, w):
#         return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

#     def _upsample_add(self, x, y):
#         '''Upsample and add two feature maps.
#         Args:
#           x: (Variable) top feature map to be upsampled.
#           y: (Variable) lateral feature map.
#         Returns:
#           (Variable) added feature map.
#         Note in PyTorch, when input size is odd, the upsampled feature map
#         with `F.upsample(..., scale_factor=2, mode='nearest')`
#         maybe not equal to the lateral feature map size.
#         e.g.
#         original input size: [N,_,15,15] ->
#         conv2d feature map size: [N,_,8,8] ->
#         upsampled feature map size: [N,_,16,16]
#         So we choose bilinear upsample which supports arbitrary output sizes.
#         '''
#         _, _, H, W = y.size()
#         return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

#     def _downsample_multiply(self, x, y):
#         return torch.mul(F.avg_pool2d(x, kernel_size=8), y)

#     def _upsample_cat(self, x, y):
#         _, _, H, W = y.size()
#         return torch.cat([F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True), y], dim=1)
    
#     def forward(self, x):
#         _, _, x_h, x_w = x.shape
#         fea_downsample = None
#         # Bottom-up
#         c1 = self.relu(self.bn1(self.conv1(x)))
#         # c1 = self.maxpool(c1)
#         c2 = self.layer1(c1)
#         c3 = self.layer2(c2)

#         if self.in_channels[2] > 0:
#             c4 = self.layer3(c3)
#             if self.out:
#                 fea_downsample = self.out(c4)
#         if self.in_channels[3] > 0:
#             c5 = self.layer4(c4)
#             if self.out:
#                 fea_downsample = self.out(c5)

#         # v2 = self.vit1(c2)
#         # # Local and global feature concatenation
#         # fea2_cat = self.fusion_mlp1(self._upsample_cat(v2, c2))  
        
#         # v3 = self.vit2(c3)
#         # # Local and global feature concatenation
#         # fea3_cat = self.fusion_mlp2(self._upsample_cat(v3, c3))  
        
#         # v4 = self.vit3(c4)
#         # # Local and global feature concatenation
#         # fea4_cat = self.fusion_mlp3(self._upsample_cat(v4, c4)) 
        
#         # print(f'c1.shape={c1.shape}, c2.shape={c2.shape}, c3.shape={c3.shape}, c4.shape={c4.shape}')
#         # c1.shape = torch.Size([4, 64, 288, 288]), c2.shape = torch.Size([4, 64, 288, 288]), c3.shape = torch.Size(
#         #     [4, 128, 144, 144]), c4.shape = torch.Size([4, 256, 144, 144])
#         # dec_4_downsample = F.interpolate(c4, size=(x_h, x_w), mode='bilinear', align_corners=True)
#         # dec_3_downsample = F.interpolate(c3, size=(x_h, x_w), mode='bilinear', align_corners=True)
#         # dec_2_downsample = F.interpolate(c2, size=(x_h, x_w), mode='bilinear', align_corners=True)
        
#         # Top-down
#         if self.in_channels[3] > 0:
#             p5 = self.toplayer(c5)
#             p4 = self._upsample_add(p5, self.latlayer1(c4))
#             p3 = self._upsample_add(p4, self.latlayer2(c3))
#             p2 = self._upsample_add(p3, self.latlayer3(c2))
#         if self.in_channels[2] > 0:
#             p4 = self.toplayer(c4)
#             p3 = self._upsample_add(p4, self.latlayer1(c3))
#             p2 = self._upsample_add(p3, self.latlayer2(c2))

        
       
#         # col_fea_up = self._upsample_cat(col_feats_batch, x_up)  # feat_down: [4, 24, 288, 288]
#         # Smooth
#         # if self.in_channels[2] > 0:
#         #     p4 = self.smooth1(p4)
#         # p3 = self.smooth2(p3)
#         # p2 = self.smooth3(p2)

#         # print(f'p2.shape={p2.shape},p3.shape={p3.shape},p4.shape={p4.shape}')
#         # p2.shape = torch.Size([4, 256, 288, 288]), p3.shape = torch.Size([4, 256, 144, 144]), p4.shape = torch.Size(
#         #     [4, 256, 144, 144])
#         # Semantic
#         # s4 =None
#         _, _, h, w = p2.size()
#         if self.in_channels[3] > 0:
#             # 256->256
#             s5 = self._upsample(F.relu(self.gn12(self.conv2(p5))), h, w)
#             # 256->256
#             s5 = self._upsample(F.relu(self.gn12(self.conv2(s5))), h, w)
#             # 256->128
#             s5 = self._upsample(F.relu(self.gn11(self.semantic_branch(s5))), h, w)
#         if self.in_channels[2] > 0:
#             # 256->256
#             s4 = self._upsample(F.relu(self.gn12(self.conv2(p4))), h, w)
#             # 256->128
#             s4 = self._upsample(F.relu(self.gn11(self.semantic_branch(s4))), h, w)

#         # 256->128
#         s4 = self._upsample(F.gelu(self.gn11(self.semantic_branch(p4))), h, w)
#         s3 = self._upsample(F.gelu(self.gn11(self.semantic_branch(p3))), h, w)
#         s2 = F.gelu(self.gn11(self.semantic_branch(p2)))
       
#         if self.in_channels[3] > 0:
#             fea_upsample = self.feature_layer(s2 + s3 + s4 + s5)
#         elif self.in_channels[2] > 0:
#             # fea_upsample = self.feature_layer(s2 + s3 )#+ s4)
#             fea_upsample = self.feature_gelu(self.feature_bn(self.feature_layer(s2 + s3 + s4)))
#         else:
#             fea_upsample = self.conv3(s2 + s3)

#         return c1 ,fea_downsample,None, c4, c3, c2, fea_upsample
#         # return c1 ,fea_downsample ,None, dec_4_downsample ,dec_3_downsample ,dec_2_downsample, fea_upsample
#         # return fea_downsample, fea_upsample
# class DEM(nn.Module):
#     """Dual Enhancement Module
#     Args:
#         channel (int):
#         block_size (list): for SPP
#         reduction(int): for GCB
#         global_type: global info type, 'spp' or 'gcb'
#     """
#     def __init__(self, channel, ratio=1./16., global_type='gcb'):
#         super(DEM, self).__init__()

#         self.global_type = global_type

#         self.local_context = nn.Sequential(
#             nn.Conv2d(channel, channel, 3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(channel)
#         )
        
#         self.add_global_context = ContextBlock(channel, ratio=ratio)
        
#         self.add_gate = self.gate_build(channel * 2, 2)
        
#         self.fusion_mlp =  nn.Sequential(
#             nn.Conv2d(channel*2, channel, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(channel),
#             nn.ReLU(),
#             # nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
#         )


#     def gate_build(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=True):
#         return nn.Sequential(
#             nn.Conv2d(in_dim, in_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
#             nn.BatchNorm2d(in_dim),
#             nn.ReLU(),
#             nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
#         )

#     def forward(self, add_info):
        
#         add_local_info = self.local_context(add_info)

#         add_global_info = self.add_global_context(add_info)
        
#         globel_local = self.fusion_mlp(torch.cat((add_local_info, add_global_info), 1))

#         # if rgb_W is not None and add_W is not None:
#         #     if up:
#         #         rgb_W = F.interpolate(rgb_W, scale_factor=2, mode='bilinear')
#         #         add_W = F.interpolate(add_W, scale_factor=2, mode='bilinear')
#         #     else:
#         #         rgb_W = F.max_pool2d(rgb_W, kernel_size=2, stride=2)
#         #         add_W = F.max_pool2d(add_W, kernel_size=2, stride=2)
            
#         #     add_W = self.add_gate(torch.cat((add_local_info, add_global_info), 1)) + add_W
#         # else:
#         #     rgb_W = self.rgb_gate(torch.cat((rgb_local_info, rgb_global_info), 1))
#         add_W = self.add_gate(torch.cat((add_local_info, add_global_info), 1))

#         normalized_add_W = F.softmax(add_W, dim=1)

#         globel_local = add_local_info * (normalized_add_W[:, 0, :, :].unsqueeze(1)) + add_global_info * (normalized_add_W[:, 1, :, :].unsqueeze(1))
#         # add_info = add_info + rgb_local_info * (normalized_rgb_W[:, 0, :, :].unsqueeze(1)) + rgb_global_info * (normalized_rgb_W[:, 1, :, :].unsqueeze(1))

#         return globel_local



# import math
# import logging
# from functools import partial
# from collections import OrderedDict
# from copy import deepcopy

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
# from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
# from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
# from timm.models.registry import register_model

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# class Block(nn.Module):

#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x


# class AgentAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
#                  agent_num=49, window=14, **kwargs):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.softmax = nn.Softmax(dim=-1)

#         self.agent_num = agent_num
#         self.window = window

#         self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
#                              padding=1, groups=dim)
#         self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
#         self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
#         self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
#         self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
#         self.ac_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1))
#         self.ca_bias = nn.Parameter(torch.zeros(1, num_heads, 1, agent_num))
#         trunc_normal_(self.an_bias, std=.02)
#         trunc_normal_(self.na_bias, std=.02)
#         trunc_normal_(self.ah_bias, std=.02)
#         trunc_normal_(self.aw_bias, std=.02)
#         trunc_normal_(self.ha_bias, std=.02)
#         trunc_normal_(self.wa_bias, std=.02)
#         trunc_normal_(self.ac_bias, std=.02)
#         trunc_normal_(self.ca_bias, std=.02)
#         pool_size = int(agent_num ** 0.5)
#         self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

#     def forward(self, x):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         b, n, c = x.shape
#         h = int(n ** 0.5)
#         w = int(n ** 0.5)
#         num_heads = self.num_heads
#         head_dim = c // num_heads
#         qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#         # q, k, v: b, n, c

#         agent_tokens = self.pool(q.reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
#         q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
#         agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

#         position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
#         position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
#         position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
#         position_bias = position_bias1 + position_bias2
#         # position_bias = torch.cat([self.ac_bias.repeat(b, 1, 1, 1), position_bias], dim=-1)
#         agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
#         agent_attn = self.attn_drop(agent_attn)
#         agent_v = agent_attn @ v

#         agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
#         agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
#         agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
#         agent_bias = agent_bias1 + agent_bias2
#         # agent_bias = torch.cat([self.ca_bias.repeat(b, 1, 1, 1), agent_bias], dim=-2)
#         q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
#         q_attn = self.attn_drop(q_attn)
#         x = q_attn @ agent_v

#         x = x.transpose(1, 2).reshape(b, n, c)
#         v_ = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
#         x[:, :, :] = x[:, :, :] + self.dwc(v_).permute(0, 2, 3, 1).reshape(b, n , c)

#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class AgentBlock(nn.Module):

#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
#                  agent_num=49, window=14):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = AgentAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
#                                    agent_num=agent_num, window=window)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x


# class VisionTransformer(nn.Module):
#     """ Vision Transformer

#     A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
#         - https://arxiv.org/abs/2010.11929

#     Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
#         - https://arxiv.org/abs/2012.12877
#     """

#     def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
#                  num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
#                  act_layer=None, weight_init='',
#                  agent_num=[49, 49, 49, 49], agent_layer=-1):
#         """
#         Args:
#             img_size (int, tuple): input image size
#             patch_size (int, tuple): patch size
#             in_chans (int): number of input channels
#             num_classes (int): number of classes for classification head
#             embed_dim (int): embedding dimension
#             depth (int): depth of transformer
#             num_heads (int): number of attention heads
#             mlp_ratio (int): ratio of mlp hidden dim to embedding dim
#             qkv_bias (bool): enable bias for qkv if True
#             representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
#             distilled (bool): model includes a distillation token and head as in DeiT models
#             drop_rate (float): dropout rate
#             attn_drop_rate (float): attention dropout rate
#             drop_path_rate (float): stochastic depth rate
#             embed_layer (nn.Module): patch embedding layer
#             norm_layer: (nn.Module): normalization layer
#             weight_init: (str): weight init scheme
#         """
#         super().__init__()
#         self.num_classes = num_classes
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#         self.num_tokens = 2 if distilled else 1
#         norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
#         act_layer = act_layer or nn.GELU
#         agent_layer = agent_layer if agent_layer > 0 else depth

#         self.patch_embed = embed_layer(
#             img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
#         num_patches = self.patch_embed.num_patches

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches , embed_dim))
#         self.pos_drop = nn.Dropout(p=drop_rate)

#         stage_num = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 3}
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
#         self.blocks = nn.Sequential(*[
#             AgentBlock(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
#                 attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
#                 agent_num=int(agent_num[stage_num[i // 2]]),
#                 window=img_size // patch_size) if i < agent_layer else
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
#                 attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
#             for i in range(depth)])
#         self.norm = norm_layer(embed_dim)

#         # Representation layer
#         if representation_size and not distilled:
#             self.num_features = representation_size
#             self.pre_logits = nn.Sequential(OrderedDict([
#                 ('fc', nn.Linear(embed_dim, representation_size)),
#                 ('act', nn.Tanh())
#             ]))
#         else:
#             self.pre_logits = nn.Identity()
            
#         # self.rearrange = Rearrange('b (h w) (p1 p2 c) -> b c h w', h = img_size, w=img_size)
#         self.rearrange = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = img_size // patch_size, w=img_size // patch_size, p1 = patch_size, p2 = patch_size)
        
#         # self.rearrange = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = temp_h, w=temp_w, p1 = patch_h_size, p2 = patch_w_size)

#         # Classifier head(s)
#         # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
#         # self.head_dist = None
#         # if distilled:
#         #     self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

#         self.init_weights(weight_init)

#     def init_weights(self, mode=''):
#         assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
#         head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
#         trunc_normal_(self.pos_embed, std=.02)
#         if self.dist_token is not None:
#             trunc_normal_(self.dist_token, std=.02)
#         if mode.startswith('jax'):
#             # leave cls token as zeros to match jax impl
#             named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
#         else:
#             trunc_normal_(self.cls_token, std=.02)
#             self.apply(_init_vit_weights)

#     def _init_weights(self, m):
#         # this fn left here for compat with downstream users
#         _init_vit_weights(m)

#     # @torch.jit.ignore()
#     # def load_pretrained(self, checkpoint_path, prefix=''):
#     #     _load_weights(self, checkpoint_path, prefix)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token', 'dist_token'}

#     def get_classifier(self):
#         if self.dist_token is None:
#             return self.head
#         else:
#             return self.head, self.head_dist

#     def reset_classifier(self, num_classes, global_pool=''):
#         self.num_classes = num_classes
#         self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
#         if self.num_tokens == 2:
#             self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

#     def forward_features(self, x):
#         x = self.patch_embed(x)
#         # cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         # if self.dist_token is None:
#         #     x = torch.cat((cls_token, x), dim=1)
#         # else:
#         #     x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
#         x = self.pos_drop(x + self.pos_embed)
#         x = self.blocks(x)
#         x = self.norm(x)
        
#         return x
#         if self.dist_token is None:
#             return self.pre_logits(x[:, 0])
#         else:
#             return x[:, 0], x[:, 1]

#     def forward(self, x):
#         x = self.forward_features(x)
        
#         x = self.rearrange(x)
#         # if self.head_dist is not None:
#         #     x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
#         #     if self.training and not torch.jit.is_scripting():
#         #         # during inference, return the average of both classifier predictions
#         #         return x, x_dist
#         #     else:
#         #         return (x + x_dist) / 2
#         # else:
#         #     x = self.head(x)
#         return x
    
# def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
#     """ ViT weight initialization
#     * When called without n, head_bias, jax_impl args it will behave exactly the same
#       as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
#     * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
#     """
#     if isinstance(module, nn.Linear):
#         if name.startswith('head'):
#             nn.init.zeros_(module.weight)
#             nn.init.constant_(module.bias, head_bias)
#         elif name.startswith('pre_logits'):
#             lecun_normal_(module.weight)
#             nn.init.zeros_(module.bias)
#         else:
#             if jax_impl:
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     if 'mlp' in name:
#                         nn.init.normal_(module.bias, std=1e-6)
#                     else:
#                         nn.init.zeros_(module.bias)
#             else:
#                 trunc_normal_(module.weight, std=.02)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#     elif jax_impl and isinstance(module, nn.Conv2d):
#         # NOTE conv was left to pytorch default in my original init
#         lecun_normal_(module.weight)
#         if module.bias is not None:
#             nn.init.zeros_(module.bias)
#     elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
#         nn.init.zeros_(module.bias)
#         nn.init.ones_(module.weight)



@MODELS.register_module("ResNet34")
class ResNetFCN(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True, config=None):
        super(ResNetFCN, self).__init__()

        if backbone == "resnet34":
            net = resnet34(pretrained)
        else:
            raise NotImplementedError("invalid backbone: {}".format(backbone))
        self.hiden_size = 128
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        #self.conv1.weight.data = net.conv1.weight.data
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        
        # self.vit1 = VisionTransformer(img_size=128,
        #                               patch_size=4,
        #                               depth=6,
        #                               num_heads=16,
        #                             #   patch_h_size=4,
        #                             #   patch_w_size=4,
        #                             in_chans=64,
        #                             embed_dim=1024,)
        # self.vit2 = VisionTransformer(img_size=64,
        #                               patch_size=2,
        #                               depth=6,
        #                               num_heads=16,
        #                     #   patch_h_size=2,
        #                     #   patch_w_size=2,
        #                             in_chans=128,
        #                             embed_dim=512,)
        # self.vit3 = VisionTransformer(img_size=32,
        #                               patch_size=1,
        #                               depth=6,
        #                               num_heads=16,
        #                     #   patch_h_size=1,
        #                     #   patch_w_size=1,
        #                       in_chans=256,
        #                       embed_dim=256,)
        # self.vit4 = VisionTransformer(img_size=16,
        #                                 patch_size=1,
        #                                 depth=6,
        #                               num_heads=16,
        #                     #   patch_h_size=1,
        #                     #   patch_w_size=1,
        #                       in_chans=512,
        #                       embed_dim=512,)
        
        channels = [64,128,256,512]
        self.fusion_mlp = nn.ModuleList()
        for i in range(4):
            self.fusion_mlp.append(nn.Sequential(
                nn.Conv2d(channels[i]*2,channels[i], kernel_size=3, stride=1, padding=1, bias=False),
                # nn.Linear(channels[i]*2,channels[i]),
                nn.BatchNorm2d(channels[i]),
                nn.ReLU(True)
                ))
                
            
        # self.gcb1 = DEM(channel = 64,ratio=1./16.)
        # self.gcb2 = DEM(channel = 128,ratio=1./16.)
        # self.gcb3 = DEM(channel = 256,ratio=1./16.)
        # self.gcb4 = DEM(channel = 512,ratio=1./16.)
        # Decoder
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(64, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(128, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        # self.gcb5 = DEM(channel = 128,ratio=1./16.)
        # self.gcb6 = DEM(channel = 128,ratio=1./16.)
        # self.gcb7 = DEM(channel = 128,ratio=1./16.)
        # self.gcb8 = DEM(channel = 128,ratio=1./16.)

        # self.distribution_pred_head = nn.Sequential(
        #     nn.Conv2d(self.hiden_size*4, self.hiden_size, kernel_size= 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.hiden_size, 8, kernel_size= 1,bias=False),
        # )

        self.multihead_distribution_classifier = nn.ModuleList()
        for i in range(4):
            self.multihead_distribution_classifier.append(
            nn.Sequential(
            nn.Conv2d(self.hiden_size, self.hiden_size, kernel_size= 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hiden_size, 8, kernel_size= 1,bias=False))
            )

    def forward(self, img, indices):
        #x = data_dict['img']
        x = img
        h, w = x.shape[2], x.shape[3]
        if h % 16 != 0 or w % 16 != 0:
            assert False, "invalid input size: {}".format(x.shape)
        
        #pdb.set_trace()

        # # Encoder
        # conv1_out = self.relu(self.bn1(self.conv1(x)))
        # layer1_out = self.layer1(self.maxpool(conv1_out))
        # globel1 = self.vit1(layer1_out)
        # gl1 = self.fusion_mlp[0](torch.cat([layer1_out,globel1],1))
        
        # layer2_out = self.layer2(gl1)
        # globel2 = self.vit2(layer2_out)
        # gl2 = self.fusion_mlp[1](torch.cat([layer2_out,globel2],1))
        
        # layer3_out = self.layer3(gl2)
        # globel3 = self.vit3(layer3_out)
        # gl3 = self.fusion_mlp[2](torch.cat([layer3_out,globel3],1))
        
        # layer4_out = self.layer4(gl3)
        # globel4 = self.vit4(layer4_out)
        # gl4 = self.fusion_mlp[3](torch.cat([layer4_out,globel4],1))

        # # Deconv
        # layer1_out = self.deconv_layer1(gl1)
        # layer2_out = self.deconv_layer2(gl2)
        # layer3_out = self.deconv_layer3(gl3)
        # layer4_out = self.deconv_layer4(gl4)
        
        conv1_out = self.relu(self.bn1(self.conv1(x)))
        layer1_out = self.layer1(self.maxpool(conv1_out))
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)

        # Deconv
        layer1_out = self.deconv_layer1(layer1_out)
        layer2_out = self.deconv_layer2(layer2_out)
        layer3_out = self.deconv_layer3(layer3_out)
        layer4_out = self.deconv_layer4(layer4_out)
        

        feat_list = [layer1_out,layer2_out,layer3_out,layer4_out]
        dist_pred_list = []
        for i in range(4):
            dist_pred = self.multihead_distribution_classifier[i](feat_list[i])
            dist_pred_list.append(dist_pred)


        # stack_feats = torch.cat([layer1_out,layer2_out,layer3_out,layer4_out],dim=1)
        # stack_feats = self.distribution_pred_head(stack_feats)
        # data_dict['img_scale2'] = layer1_out
        # data_dict['img_scale4'] = layer2_out
        # data_dict['img_scale8'] = layer3_out
        # data_dict['img_scale16'] = layer4_out

        #process_keys = [k for k in data_dict.keys() if k.find('img_scale') != -1]
        img_indices = indices

        temp = [[] for k in range(len(feat_list))]
       

        for i in range(x.shape[0]):
            for k in range(len(feat_list)):
                ss = feat_list[k].permute(0, 2, 3, 1).contiguous()
                temp[k].append(ss[i][img_indices[i][:, 1], img_indices[i][:, 0]].contiguous())

        for k in range(len(feat_list)):
            feat_list[k] = torch.cat(temp[k], 0).contiguous()

        return feat_list, dist_pred_list
    

