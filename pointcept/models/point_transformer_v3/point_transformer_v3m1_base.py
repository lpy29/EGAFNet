"""
Point Transformer - V3 Mode1

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath
import torch.nn.functional as F

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential

class FFN(nn.Module):

    def __init__(self, d_model = 64, hidden_dim = 1024, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.net(x)
        output = output + x
        output = self.norm(output)
        return output
    
class SelfAttention(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, pe=None):
        """
        x Tensor (b, 100, c)
        """
        q = k = self.with_pos_embed(x, pe).unsqueeze(0)
        x = x.unsqueeze(0)
        output, _ = self.attn(q, k, x)
        output = self.dropout(output) + x
        output = self.norm(output)
        return output
    
    
class CrossAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8,dropout_ratio = 0.0):
        super(CrossAttention, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout_ratio,batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_ratio)


    def forward(self, query, key):
        
        q = query.unsqueeze(0)  # (1, n, d_model)
            
        B, C , W, H = key.shape
        # 使用 permute 将 Tensor 转换为 (B, W, H, C) view 将 Tensor 重塑为 (B, W*H, C)
        k = v = key.permute(0, 2, 3, 1).view(B, W * H, C)
        output, _ = self.attention(q, k, v)
            
        # self.dropout(output)
        # Add & Norm
        output = q + output
        output = self.norm(output)
       
        
        return output


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


@MODELS.register_module("PT-v3m1")
class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")
        
        self.fcs1 = nn.ModuleList()
        # self.fcs2 = nn.ModuleList()
        for i in range(4):
            self.fcs1.append(nn.Sequential(
                nn.Linear(128, enc_channels[i+1]),
                nn.BatchNorm1d(enc_channels[i+1]),
                nn.ReLU(True)
                )
            )
            # self.fcs2.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size))) 
        # self.fusion_mlp_dec0 = nn.Linear(64+enc_channels[0], enc_channels[0])
        # self.fusion_mlp_dec1 = nn.Linear(64+enc_channels[1], enc_channels[1])
        # self.fusion_mlp_dec2 = nn.Linear(128+enc_channels[2], enc_channels[2])
        # self.fusion_mlp_dec3 = nn.Linear(256+enc_channels[3], enc_channels[3])
        # self.fusion_mlp_dec4 = nn.Linear(512+enc_channels[4], enc_channels[4])
        # self.fusion_mlp_dec5 = nn.Linear(256+enc_channels[-1], enc_channels[-1])
        
       # change the dsm feature to pc feature space
        self.fusion_mlp_dec0 = nn.Sequential(
                nn.Linear(64, enc_channels[0]),
                nn.BatchNorm1d(enc_channels[0]),
                nn.ReLU(True),
                )
        
        self.fusion_mlp_0 = nn.Sequential(
                nn.Linear(64, enc_channels[0]),
                nn.BatchNorm1d(enc_channels[0]),
                nn.ReLU(True),
                )
        
        # change z to another channels
        self.fusion_z = nn.Sequential(
                nn.Linear(3, enc_channels[0]),
                nn.BatchNorm1d(enc_channels[0]),
                nn.ReLU(True),
                )
        # calculate the weight
        self.z_adaptor_0 = (
            nn.Sequential(
                nn.Linear(enc_channels[0]*2, enc_channels[0]),
                nn.Sigmoid()
                )
        )
        
        
        self.fusion_mlp_dec1 = nn.Sequential(
                nn.Linear(64, enc_channels[0]),
                nn.BatchNorm1d(enc_channels[0]),
                nn.ReLU(True),
                )
        
        self.fusion_mlp_1 = nn.Sequential(
                nn.Linear(64, enc_channels[0]),
                nn.BatchNorm1d(enc_channels[0]),
                nn.ReLU(True),
                )
    
        self.fusion_z_1 = nn.Sequential(
                nn.Linear(3, enc_channels[0]),
                nn.BatchNorm1d(enc_channels[0]),
                nn.ReLU(True),
                )
        self.z_adaptor_1 = (
            nn.Sequential(
                nn.Linear(enc_channels[0]*2, 1),
                nn.Sigmoid()
                )
        )
        
        
        
        self.fusion_mlp_dec2 = nn.Sequential(
                nn.Linear(128, enc_channels[1]),
                nn.BatchNorm1d(enc_channels[1]),
                nn.ReLU(True),
                )
        self.fusion_mlp_2 = nn.Sequential(
                nn.Linear(128, enc_channels[1]),
                nn.BatchNorm1d(enc_channels[1]),
                nn.ReLU(True),
                )

        self.fusion_z_2 = nn.Sequential(
                nn.Linear(3, enc_channels[1]),
                nn.BatchNorm1d(enc_channels[1]),
                nn.ReLU(True),
                )
        self.z_adaptor_2 = (
            nn.Sequential(
                nn.Linear(enc_channels[1]*2, 1),
                nn.Sigmoid()
                )
        )
        
        self.fusion_mlp_dec3 = nn.Sequential(
                nn.Linear(256, enc_channels[2]),
                nn.BatchNorm1d(enc_channels[2]),
                nn.ReLU(True),
                )
        
        self.fusion_mlp_3 = nn.Sequential(
                nn.Linear(256, enc_channels[2]),
                nn.BatchNorm1d(enc_channels[2]),
                nn.ReLU(True),
                )

        self.fusion_z_3 = nn.Sequential(
                nn.Linear(3, enc_channels[2]),
                nn.BatchNorm1d(enc_channels[2]),
                nn.ReLU(True),
                )
        self.z_adaptor_3 = (
            nn.Sequential(
                nn.Linear(enc_channels[2]*2, 1),
                nn.Sigmoid()
                )
        )
        
        self.fusion_mlp_dec4 = nn.Sequential(
                nn.Linear(512, enc_channels[3]),
                nn.BatchNorm1d(enc_channels[3]),
                # nn.GELU()
                nn.ReLU(True),
                )
        self.fusion_mlp_4 = nn.Sequential(
                nn.Linear(512, enc_channels[3]),
                nn.BatchNorm1d(enc_channels[3]),
                # nn.GELU()
                nn.ReLU(True),
                )

        self.fusion_z_4 = nn.Sequential(
                nn.Linear(3, 128),
                nn.BatchNorm1d(128),
                # nn.GELU()
                nn.ReLU(True),
                )
        self.z_adaptor_4 = (
            nn.Sequential(
                nn.Linear(enc_channels[3]+128,enc_channels[3]),
                nn.Sigmoid()
                )
        )
        
        # self.fusion_mlp_dec2 = nn.Linear(128, enc_channels[2])
        # self.fusion_mlp_2 = nn.Linear(enc_channels[2]*2, enc_channels[2])
        # self.fusion_mlp_dec3 = nn.Linear(256+enc_channels[2], enc_channels[2])
        # self.fusion_mlp_dec4 = nn.Linear(256+enc_channels[3], enc_channels[3])
        # self.fusion_mlp_dec5 = nn.Linear(128, enc_channels[-1])
        # self.fusion_mlp_5 = nn.Linear(enc_channels[-1]*2, enc_channels[-1])    
        # self.fusion_z_5 = nn.Sequential(
        #         nn.Linear(3, 128),
        #         nn.BatchNorm1d(enc_channels[0]),
        #         nn.ReLU(True),
        #         )
        
        
        # # # self.fusion_mlp_dec0 = nn.Conv2d(64+enc_channels[0], enc_channels[0],kernel_size=1)
        # # # self.fusion_mlp_dec1 = nn.Conv2d(64+enc_channels[1], enc_channels[1],kernel_size=1)
        # # # self.fusion_mlp_dec2 = nn.Conv2d(128+enc_channels[2], enc_channels[2],kernel_size=1)
        # # # self.fusion_mlp_dec3 = nn.Linear(256+enc_channels[2], enc_channels[2])
        # # # self.fusion_mlp_dec4 = nn.Linear(256+enc_channels[3], enc_channels[3])
        # # # self.fusion_mlp_dec5 = nn.Linear(256+enc_channels[-1], enc_channels[-1])
        
        # self.z_adaptor_0 = (
        #     nn.Sequential(
        #         nn.Linear(enc_channels[0]*2, 1),
        #         nn.Sigmoid()
        #         )
        # )
        # self.z_adaptor_1 = (
        #     nn.Sequential(
        #         nn.Linear(enc_channels[0]+128, 1),
        #         nn.Sigmoid()
        #         )
        # )
        # # self.z_adaptor_2 = (
        # #     nn.Sequential(
        # #         nn.Linear(enc_channels[2]+3, 64),
        # #         nn.BatchNorm1d(64),
        # #         nn.ReLU(True),
        # #         nn.Linear(64, 1),
        # #         nn.Sigmoid()
        # #         )
        # # )
        
        # self.z_adaptor_5 = (
        #     nn.Sequential(
        #         nn.Linear(enc_channels[-1]+128, 1),
        #         nn.BatchNorm1d(64),
        #         nn.ReLU(True),
        #         nn.Linear(64, 1),
        #         nn.Sigmoid()
        #         )
        # )
                
    def fuse_features(self, point, zstd ,image, image_feat, fusion_mlp_dec,fusion_mlp,fusion_z,z_adaptor):
        # Add fusion
        batch_offset = point.offset
        batch_size = len(batch_offset) 
        batch_offset = torch.cat((torch.tensor([0]).cuda(),batch_offset))  # 添加结束索引
        
        
        dsm_feats = []
        dsm_zs = []
        dsm_stds = []
        point_zs = []
        
        for i in range(batch_size):
            start_id = batch_offset[i]
            end_id = batch_offset[i + 1]
            
            proj_ = point.grid_coord[start_id:end_id,:2]
            W = proj_[:,0].max() +1#- proj_[:,0].min() + 1  
            H = proj_[:,1].max() +1#- proj_[:,1].min() + 1
            img_feat = nn.functional.interpolate(image_feat[i], size=(W,H), mode='bilinear', align_corners=True).squeeze(0).permute(1,2,0)
            img = nn.functional.interpolate(torch.tensor(image[i]).cuda().unsqueeze(0).unsqueeze(0), size=(W,H), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
            std = nn.functional.interpolate(torch.tensor(zstd[i]).cuda().unsqueeze(0).unsqueeze(0), size=(W,H), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
        

            # img_feat = image_feat[i].squeeze(0).permute(1,2,0)
            # pc_feat = point.feat[start_id:end_id].clone()
            
            # get img_feat & dsm_features
            ixs, iys = proj_[:, 0], proj_[:, 1]
            dsm_z = img[ixs,iys].float().cuda()
            dsm_zstd = std[ixs,iys].float().cuda()

            # std norm
            # std_min = torch.min(dsm_zstd)
            # std_max = torch.max(dsm_zstd)
            # z_score_variances = (dsm_zstd - std_min) / (std_max - std_min)
            # z_score_variances = torch.clamp(z_score_variances, min=1e-8, max=0.99)  # 防止数值溢出

            
            dsm_feat = img_feat[ixs,iys]
            # dsm_feat = dsm_feat / torch.norm(dsm_feat,dim=-1,keepdim=True)
            
            point_z = point.coord[start_id:end_id,2].reshape(-1,1)
            point_zmin = torch.min(point_z)
            point_zmax = torch.max(point_z)
            point_zmean = torch.mean(point_z)
            point_zstd = torch.std(point_z)
            z_norm = (point_z-point_zmean)/point_zstd
            # z_norm = (point_z-point_zmin)/(point_zmax-point_zmin)
            # z_norm = torch.clamp(z_norm, min=1e-8, max=0.99)  # 防止数值溢出

            # std_mean = torch.mean(dsm_zstd)
            # std_std = torch.std(dsm_zstd)
            # zstd_norm = (dsm_zstd-std_mean)/std_std
            
            dsm_zs.append(dsm_z)
            dsm_stds.append(dsm_zstd)
            dsm_feats.append(dsm_feat)
            point_zs.append(z_norm)
        
        dsm_feats = torch.cat((dsm_feats),dim=0)
        dsm_zs = torch.cat((dsm_zs),dim=0).reshape(-1,1)
        dsm_stds = torch.cat((dsm_stds),dim=0).reshape(-1,1)
        
        # point_zmean = torch.mean(point.coord[:,2])
        # point_zstd = torch.std(point.coord[:,2])
        # point_zs = (point.coord[:,2]-point_zmean)/point_zstd
        point_zs = torch.cat((point_zs),dim=0).reshape(-1,1)
        
        dsm_feats = fusion_mlp(dsm_feats)
        
        # # change the channels to z related feat
        # z = fusion_z(torch.cat((point_zs,dsm_zs,dsm_stds),dim=-1))
        # # concat the z-related feat and point feat to get weight
        # w = z_adaptor(torch.cat((z,point.feat),dim=-1))
        # # weighted add 
        # fused =(1-w)*point.feat + w*dsm_feats
        
        fused_feats = torch.cat((dsm_feats,point.feat),dim=-1)
        fused = fusion_mlp_dec(fused_feats)
        return fused


    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        # point.feat = self.fuse_features(point,zstd ,image, img_embed, self.fusion_mlp_dec0, self.fusion_mlp_0, self.fusion_z,self.z_adaptor_0)
        # point = self.enc(point)
        point0 = self.enc.enc0(point)
        point1 = self.enc.enc1(point0)
        # point.feat = self.fcs1[0](torch.cat((feat_2d[0],point.feat),dim=-1))#self.fuse_features(point, zstd ,image,layer_1_upsampled, self.fusion_mlp_dec1,self.fusion_mlp_1,self.fusion_z_1,self.z_adaptor_1)
        point2 = self.enc.enc2(point1)
        # point.feat = self.fcs1[1](torch.cat((feat_2d[1],point.feat),dim=-1))#self.fuse_features(point, zstd ,image,layer_2_upsampled, self.fusion_mlp_dec2,self.fusion_mlp_2,self.fusion_z_2,self.z_adaptor_2)
        point3 = self.enc.enc3(point2)
        # point.feat = self.fcs1[2](torch.cat((feat_2d[2],point.feat),dim=-1))#self.fuse_features(point, zstd, image,layer_3_upsampled, self.fusion_mlp_dec3,self.fusion_mlp_3,self.fusion_z_3,self.z_adaptor_3)
        # point.feat = self.fuse_features(point, zstd ,image,layer_4_upsampled, self.fusion_mlp_dec4,self.fusion_mlp_4,self.fusion_z_4,self.z_adaptor_4)
        point4 = self.enc.enc4(point3)
        # point.feat = self.fuse_features(point, zstd, image,layer_2_upsampled, self.fusion_mlp_dec4,self.fusion_z_4,self.z_adaptor_4)
        # point.feat = self.fcs1[3](torch.cat((feat_2d[3],point.feat),dim=-1))#self.fuse_features(point, zstd ,image,layer_4_upsampled, self.fusion_mlp_dec4,self.fusion_z_4,self.z_adaptor_4)
        # # point = self.enc.enc4(point)
        # # point.feat = self.fuse_features(point,layer_4_upsampled, zstd ,image, proj,self.fusion_mlp_dec5)
        return  point1.feat,point2.feat,point3.feat,point4.feat
        if not self.cls_mode:
            point = self.dec(point)
        # else:
        #     point.feat = torch_scatter.segment_csr(
        #         src=point.feat,
        #         indptr=nn.functional.pad(point.offset, (1, 0)),
        #         reduce="mean",
        #     )
        return point