"""Model factory for baseline training."""
from __future__ import annotations

# LightGBMを追加
import lightgbm as lgb

# numpyを追加
import numpy as np

import logging
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple

try:
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    timm = None
    torch = None
    nn = None
    F = None

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, KFold


@dataclass
class ModelConfig:
    type: str
    params: Dict[str, Any]
    fit_intercept: bool = True


# LightGBMに適応
def create_model(config: ModelConfig):
    if config.type == "ridge":
        return Ridge(**config.params)
    if config.type == "linear_regression":
        return LinearRegression(fit_intercept=config.fit_intercept)
    # LightGBMのサポートを追加
    if config.type == "lightgbm":
        # best_paramsにはfit_interceptは含まれないため、fit_intercept以外のパラメータを渡す
        model_params = config.params.copy()
        model_params.pop("fit_intercept", None)
        return lgb.LGBMRegressor(**model_params)
    raise ValueError(f"Unsupported model type: {config.type}")


def build_cv(strategy: str, n_splits: int, shuffle: bool = False, random_state: int | None = None):
    if strategy == "time_series":
        return TimeSeriesSplit(n_splits=n_splits)
    if strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    raise ValueError(f"Unsupported CV strategy: {strategy}")


def rmse(y_true, y_pred):
    # MSEを計算し、その平方根を返す
    return np.sqrt(mean_squared_error(y_true, y_pred))

DINO_LOGGER = logging.getLogger(__name__ + ".crosspvt")
if not DINO_LOGGER.handlers:
    DINO_LOGGER.addHandler(logging.StreamHandler())
DINO_LOGGER.setLevel(logging.INFO)


@dataclass
class TrainCFG:
    dropout: float = 0.1
    hidden_ratio: float = 0.35
    # dino_candidates: Tuple[str, ...] = (
    #     "vit_base_patch14_dinov2",
    #     "vit_base_patch14_reg4_dinov2",
    #     "vit_small_patch14_dinov2",
    # )    
    dino_candidates: Tuple[str, ...] = (
        "vit_small_patch14_dinov2",
    )
    small_grid: Tuple[int, int] = (4, 4)
    big_grid: Tuple[int, int] = (2, 2)
    t2t_depth: int = 2
    cross_layers: int = 2
    cross_heads: int = 6
    pyramid_dims: Tuple[int, int, int] = (384, 512, 640)
    mobilevit_heads: int = 4
    mobilevit_depth: int = 2
    sra_heads: int = 8
    sra_ratio: int = 2
    mamba_depth: int = 3
    mamba_kernel: int = 5
    aux_head: bool = True
    aux_loss_weight: float = 0.4
    ALL_TARGET_COLS: Tuple[str, ...] = (
        "Dry_Green_g",
        "Dry_Dead_g",
        "Dry_Clover_g",
        "GDM_g",
        "Dry_Total_g",
    )


CFG = TrainCFG()


def update_cfg_from_checkpoint(cfg_dict: dict):
    """Overwrite CFG fields from a checkpoint dictionary."""
    global CFG
    if not cfg_dict:
        return
    for key, value in cfg_dict.items():
        if hasattr(CFG, key):
            setattr(CFG, key, value)


if torch is not None and nn is not None and F is not None and timm is not None:
    class FeedForward(nn.Module):
        def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
            super().__init__()
            hid = int(dim * mlp_ratio)
            self.net = nn.Sequential(
                nn.Linear(dim, hid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hid, dim),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.net(x)


    class AttentionBlock(nn.Module):
        def __init__(self, dim, heads=8, dropout=0.0, mlp_ratio=4.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.ff = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

        def forward(self, x):
            h = self.norm1(x)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + attn_out
            x = x + self.ff(self.norm2(x))
            return x


    class MobileViTBlock(nn.Module):
        """Lightweight MobileViT: local CNN + tiny Transformer."""

        def __init__(self, dim, heads=4, depth=2, patch=(2, 2), dropout=0.0):
            super().__init__()
            self.local = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                nn.Conv2d(dim, dim, 1),
                nn.GELU(),
            )
            self.patch = patch
            self.transformer = nn.ModuleList(
                [AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(depth)]
            )
            self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)

        def forward(self, x: torch.Tensor):
            local_feat = self.local(x)
            B, C, H, W = local_feat.shape
            ph, pw = self.patch
            new_h = math.ceil(H / ph) * ph
            new_w = math.ceil(W / pw) * pw
            if new_h != H or new_w != W:
                local_feat = F.interpolate(local_feat, size=(new_h, new_w), mode="bilinear", align_corners=False)
                H, W = new_h, new_w

            tokens = local_feat.unfold(2, ph, ph).unfold(3, pw, pw)
            tokens = tokens.contiguous().view(B, C, -1, ph, pw)
            tokens = tokens.permute(0, 2, 3, 4, 1).reshape(B, -1, C)

            for blk in self.transformer:
                tokens = blk(tokens)

            feat = tokens.view(B, -1, ph * pw, C).permute(0, 3, 1, 2)
            nh = H // ph
            nw = W // pw
            feat = feat.view(B, C, nh, nw, ph, pw).permute(0, 1, 2, 4, 3, 5)
            feat = feat.reshape(B, C, H, W)

            if feat.shape[-2:] != x.shape[-2:]:
                feat = F.interpolate(feat, size=x.shape[-2:], mode="bilinear", align_corners=False)

            out = self.fuse(torch.cat([x, feat], dim=1))
            return out


    class SpatialReductionAttention(nn.Module):
        def __init__(self, dim, heads=8, sr_ratio=2, dropout=0.0):
            super().__init__()
            self.heads = heads
            self.scale = (dim // heads) ** -0.5
            self.q = nn.Linear(dim, dim)
            self.kv = nn.Linear(dim, dim * 2)
            self.sr_ratio = sr_ratio
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
            else:
                self.sr = None
            self.proj = nn.Linear(dim, dim)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, hw: Tuple[int, int]):
            B, N, C = x.shape
            q = self.q(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)

            if self.sr is not None:
                H, W = hw
                feat = x.transpose(1, 2).reshape(B, C, H, W)
                feat = self.sr(feat)
                feat = feat.reshape(B, C, -1).transpose(1, 2)
                feat = self.norm(feat)
            else:
                feat = x

            kv = self.kv(feat)
            k, v = kv.chunk(2, dim=-1)
            k = k.reshape(B, -1, self.heads, C // self.heads).permute(0, 2, 3, 1)
            v = v.reshape(B, -1, self.heads, C // self.heads).permute(0, 2, 1, 3)

            attn = torch.matmul(q, k) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.drop(attn)
            out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, C)
            out = self.proj(out)
            return out


    class PVTBlock(nn.Module):
        def __init__(self, dim, heads=8, sr_ratio=2, dropout=0.0, mlp_ratio=4.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.sra = SpatialReductionAttention(dim, heads=heads, sr_ratio=sr_ratio, dropout=dropout)
            self.norm2 = nn.LayerNorm(dim)
            self.ff = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

        def forward(self, x, hw: Tuple[int, int]):
            x = x + self.sra(self.norm1(x), hw)
            x = x + self.ff(self.norm2(x))
            return x


    class LocalMambaBlock(nn.Module):
        """Simplified local Mamba block with depthwise conv and gating."""

        def __init__(self, dim, kernel_size=5, dropout=0.0):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
            self.gate = nn.Linear(dim, dim)
            self.proj = nn.Linear(dim, dim)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            shortcut = x
            x = self.norm(x)
            g = torch.sigmoid(self.gate(x))
            x = (x * g).transpose(1, 2)
            x = self.dwconv(x).transpose(1, 2)
            x = self.proj(x)
            x = self.drop(x)
            return shortcut + x


    class T2TRetokenizer(nn.Module):
        """Locally retokenize small-grid tiles and downsample to 2x2."""

        def __init__(self, dim, depth=2, heads=4, dropout=0.0):
            super().__init__()
            self.blocks = nn.ModuleList(
                [AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(depth)]
            )

        def forward(self, tokens: torch.Tensor, grid_hw: Tuple[int, int]):
            B, T, C = tokens.shape
            H, W = grid_hw
            feat_map = tokens.transpose(1, 2).reshape(B, C, H, W)
            seq = feat_map.flatten(2).transpose(1, 2)
            for blk in self.blocks:
                seq = blk(seq)
            seq_map = seq.transpose(1, 2).reshape(B, C, H, W)
            pooled = F.adaptive_avg_pool2d(seq_map, (2, 2))
            retokens = pooled.flatten(2).transpose(1, 2)
            return retokens, seq_map


    class CrossScaleFusion(nn.Module):
        def __init__(self, dim, heads=6, dropout=0.0, layers=2):
            super().__init__()
            self.layers_s = nn.ModuleList(
                [AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(layers)]
            )
            self.layers_b = nn.ModuleList(
                [AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(layers)]
            )
            self.cross_s = nn.ModuleList(
                [
                    nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, kdim=dim, vdim=dim)
                    for _ in range(layers)
                ]
            )
            self.cross_b = nn.ModuleList(
                [
                    nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, kdim=dim, vdim=dim)
                    for _ in range(layers)
                ]
            )
            self.norm_s = nn.LayerNorm(dim)
            self.norm_b = nn.LayerNorm(dim)

        def forward(self, tok_s: torch.Tensor, tok_b: torch.Tensor):
            B, Ts, C = tok_s.shape
            cls_s = tok_s.new_zeros(B, 1, C)
            cls_b = tok_b.new_zeros(B, 1, C)
            tok_s = torch.cat([cls_s, tok_s], dim=1)
            tok_b = torch.cat([cls_b, tok_b], dim=1)

            for ls, lb, cs, cb in zip(self.layers_s, self.layers_b, self.cross_s, self.cross_b):
                tok_s = ls(tok_s)
                tok_b = lb(tok_b)
                q_s = self.norm_s(tok_s[:, :1])
                q_b = self.norm_b(tok_b[:, :1])
                cls_s_upd, _ = cs(
                    q_s,
                    torch.cat([tok_b, q_b], dim=1),
                    torch.cat([tok_b, q_b], dim=1),
                    need_weights=False,
                )
                cls_b_upd, _ = cb(
                    q_b,
                    torch.cat([tok_s, q_s], dim=1),
                    torch.cat([tok_s, q_s], dim=1),
                    need_weights=False,
                )
                tok_s = torch.cat([tok_s[:, :1] + cls_s_upd, tok_s[:, 1:]], dim=1)
                tok_b = torch.cat([tok_b[:, :1] + cls_b_upd, tok_b[:, 1:]], dim=1)

            tokens = torch.cat([tok_s[:, :1], tok_b[:, :1], tok_s[:, 1:], tok_b[:, 1:]], dim=1)
            return tokens


    class TileEncoder(nn.Module):
        def __init__(self, backbone: nn.Module, input_res: int):
            super().__init__()
            self.backbone = backbone
            self.input_res = input_res

        def forward(self, x: torch.Tensor, grid: Tuple[int, int]):
            B, C, H, W = x.shape
            r, c = grid
            hs = torch.linspace(0, H, steps=r + 1, device=x.device).round().long()
            ws = torch.linspace(0, W, steps=c + 1, device=x.device).round().long()
            tiles = []
            for i in range(r):
                for j in range(c):
                    rs, re = hs[i].item(), hs[i + 1].item()
                    cs, ce = ws[j].item(), ws[j + 1].item()
                    xt = x[:, :, rs:re, cs:ce]
                    if xt.shape[-2:] != (self.input_res, self.input_res):
                        xt = F.interpolate(
                            xt, size=(self.input_res, self.input_res), mode="bilinear", align_corners=False
                        )
                    tiles.append(xt)
            tiles = torch.stack(tiles, dim=1)
            flat = tiles.view(-1, C, self.input_res, self.input_res)
            feats = self.backbone(flat)
            feats = feats.view(B, -1, feats.shape[-1])
            return feats


    class PyramidMixer(nn.Module):
        def __init__(
            self,
            dim_in: int,
            dims: Tuple[int, int, int],
            mobilevit_heads: int = 4,
            mobilevit_depth: int = 2,
            sra_heads: int = 6,
            sra_ratio: int = 2,
            mamba_depth: int = 3,
            mamba_kernel: int = 5,
            dropout: float = 0.0,
        ):
            super().__init__()
            c1, c2, c3 = dims
            self.proj1 = nn.Linear(dim_in, c1)
            self.mobilevit = MobileViTBlock(c1, heads=mobilevit_heads, depth=mobilevit_depth, dropout=dropout)
            self.proj2 = nn.Linear(c1, c2)
            self.pvt = PVTBlock(c2, heads=sra_heads, sr_ratio=sra_ratio, dropout=dropout, mlp_ratio=3.0)
            self.mamba_local = LocalMambaBlock(c2, kernel_size=mamba_kernel, dropout=dropout)
            self.proj3 = nn.Linear(c2, c3)
            self.mamba_global = nn.ModuleList(
                [LocalMambaBlock(c3, kernel_size=mamba_kernel, dropout=dropout) for _ in range(mamba_depth)]
            )
            self.final_attn = AttentionBlock(c3, heads=min(8, c3 // 64 + 1), dropout=dropout, mlp_ratio=2.0)

        def _tokens_to_map(self, tokens: torch.Tensor, target_hw: Tuple[int, int]):
            B, N, C = tokens.shape
            H, W = target_hw
            need = H * W
            if N < need:
                pad = tokens.new_zeros(B, need - N, C)
                tokens = torch.cat([tokens, pad], dim=1)
            tokens = tokens[:, :need, :]
            feat_map = tokens.transpose(1, 2).reshape(B, C, H, W)
            return feat_map

        @staticmethod
        def _fit_hw(n_tokens: int) -> Tuple[int, int]:
            h = int(math.sqrt(n_tokens))
            w = h
            while h * w < n_tokens:
                w += 1
                if h * w < n_tokens:
                    h += 1
            return h, w

        def forward(self, tokens: torch.Tensor):
            B, N, C = tokens.shape
            map_hw = (3, 4)
            feat_map = self._tokens_to_map(tokens, map_hw)

            t1 = self.proj1(tokens)
            m1 = self._tokens_to_map(t1, map_hw)
            m1 = self.mobilevit(m1)
            t1_out = m1.flatten(2).transpose(1, 2)[:, :N]

            t2 = self.proj2(t1_out)
            new_len = max(4, N // 2)
            t2 = t2[:, :new_len] + F.adaptive_avg_pool1d(t2.transpose(1, 2), new_len).transpose(1, 2)
            hw2 = self._fit_hw(t2.size(1))
            if t2.size(1) < hw2[0] * hw2[1]:
                pad = t2.new_zeros(B, hw2[0] * hw2[1] - t2.size(1), t2.size(2))
                t2 = torch.cat([t2, pad], dim=1)
            t2 = self.pvt(t2, hw2)
            t2 = self.mamba_local(t2)

            t3 = self.proj3(t2)
            pooled = torch.stack([t3.mean(dim=1), t3.max(dim=1).values], dim=1)
            t3 = pooled
            for blk in self.mamba_global:
                t3 = blk(t3)
            t3 = self.final_attn(t3)
            global_feat = t3.mean(dim=1)
            return global_feat, {"stage1_map": m1.detach(), "stage2_tokens": t2.detach(), "stage3_tokens": t3.detach()}


    class CrossPVT_T2T_MambaDINO(nn.Module):
        def __init__(self, dropout: float = 0.1, hidden_ratio: float = 0.35):
            super().__init__()
            self.backbone, self.feat_dim, self.backbone_name, self.input_res = self._build_dino_backbone()
            self.tile_encoder = TileEncoder(self.backbone, self.input_res)
            self.t2t = T2TRetokenizer(self.feat_dim, depth=CFG.t2t_depth, heads=CFG.cross_heads, dropout=dropout)
            self.cross = CrossScaleFusion(
                self.feat_dim, heads=CFG.cross_heads, dropout=dropout, layers=CFG.cross_layers
            )
            self.pyramid = PyramidMixer(
                dim_in=self.feat_dim,
                dims=CFG.pyramid_dims,
                mobilevit_heads=CFG.mobilevit_heads,
                mobilevit_depth=CFG.mobilevit_depth,
                sra_heads=CFG.sra_heads,
                sra_ratio=CFG.sra_ratio,
                mamba_depth=CFG.mamba_depth,
                mamba_kernel=CFG.mamba_kernel,
                dropout=dropout,
            )

            combined = CFG.pyramid_dims[-1] * 2
            self.combined_dim = combined
            hidden = max(32, int(combined * hidden_ratio))

            def head():
                return nn.Sequential(
                    nn.Linear(combined, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )

            self.head_green = head()
            self.head_clover = head()
            self.head_dead = head()
            self.score_head = nn.Sequential(nn.LayerNorm(combined), nn.Linear(combined, 1))
            self.aux_head = (
                nn.Sequential(nn.LayerNorm(CFG.pyramid_dims[1]), nn.Linear(CFG.pyramid_dims[1], 5))
                if CFG.aux_head
                else None
            )
            self.softplus = nn.Softplus(beta=1.0)

            self.cross_gate_left = nn.Linear(CFG.pyramid_dims[-1], CFG.pyramid_dims[-1])
            self.cross_gate_right = nn.Linear(CFG.pyramid_dims[-1], CFG.pyramid_dims[-1])

        def summary(self) -> str:
            def count_params(m):
                total = sum(p.numel() for p in m.parameters())
                trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
                return total, trainable

            total_params, trainable_params = count_params(self)
            summary_lines = [
                f"Model: {self.__class__.__name__}",
                f"Backbone: {self.backbone_name}",
                f"Input resolution: {self.input_res}",
                f"Total parameters: {total_params:,}",
                f"Trainable parameters: {trainable_params:,}",
                "",
                "TrainCFG:",
            ]
            cfg_dict = asdict(CFG)
            for key in sorted(cfg_dict.keys()):
                summary_lines.append(f"  {key}: {cfg_dict[key]}")
            return "\n".join(summary_lines)

        def _build_dino_backbone(self):
            last_err = None
            for name in CFG.dino_candidates:
                for gp in ["token", "avg", "__default__"]:
                    try:
                        if gp == "__default__":
                            m = timm.create_model(name, pretrained=False, num_classes=0)
                            gp_str = "default"
                        else:
                            m = timm.create_model(name, pretrained=False, num_classes=0, global_pool=gp)
                            gp_str = gp
                        feat = m.num_features
                        input_res = self._infer_input_res(m)
                        DINO_LOGGER.info(
                            f"Using DINO backbone: {name} | global_pool={gp_str} | feat_dim={feat} | input_res={input_res}"
                        )
                        if hasattr(m, "set_grad_checkpointing"):
                            m.set_grad_checkpointing(True)
                        return m, feat, name, int(input_res)
                    except Exception as err:
                        last_err = err
                        continue
            raise RuntimeError(f"Failed to create any DINO backbone. Last error: {last_err}")

        @staticmethod
        def _infer_input_res(model) -> int:
            if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "img_size"):
                isz = model.patch_embed.img_size
                return int(isz if isinstance(isz, (int, float)) else isz[0])
            if hasattr(model, "img_size"):
                isz = model.img_size
                return int(isz if isinstance(isz, (int, float)) else isz[0])
            default_cfg = getattr(model, "default_cfg", {}) or {}
            input_size = default_cfg.get("input_size", None)
            if input_size:
                if isinstance(input_size, (tuple, list)) and len(input_size) >= 2:
                    return int(input_size[1])
                return int(input_size if isinstance(input_size, (int, float)) else 224)
            return 518

        def _half_forward(self, x_half: torch.Tensor):
            tiles_small = self.tile_encoder(x_half, CFG.small_grid)
            tiles_big = self.tile_encoder(x_half, CFG.big_grid)
            t2, stage1_map = self.t2t(tiles_small, CFG.small_grid)
            fused = self.cross(t2, tiles_big)
            feat, feat_maps = self.pyramid(fused)
            feat_maps["stage1_map"] = stage1_map
            return feat, feat_maps

        def _merge_heads(self, f_l: torch.Tensor, f_r: torch.Tensor):
            g_l = torch.sigmoid(self.cross_gate_left(f_r))
            g_r = torch.sigmoid(self.cross_gate_right(f_l))
            f_l = f_l * g_l
            f_r = f_r * g_r
            f = torch.cat([f_l, f_r], dim=1)
            green_pos = self.softplus(self.head_green(f))
            clover_pos = self.softplus(self.head_clover(f))
            dead_pos = self.softplus(self.head_dead(f))
            gdm = green_pos + clover_pos
            total = gdm + dead_pos
            return total, gdm, green_pos, f

        def _param_device_dtype(self):
            try:
                ref = next(self.parameters())
                return ref.device, ref.dtype
            except StopIteration:
                return torch.device("cpu"), torch.float32

        def _empty_forward_output(self, device=None, dtype=None, return_features: bool = False):
            if device is None or dtype is None:
                device_p, dtype_p = self._param_device_dtype()
                if device is None:
                    device = device_p
                if dtype is None:
                    dtype = dtype_p

            zero = torch.zeros(0, 1, device=device, dtype=dtype)
            out = {
                "total": zero,
                "gdm": zero,
                "green": zero,
                "score_feat": torch.zeros(0, self.combined_dim, device=device, dtype=dtype),
            }
            if self.aux_head is not None:
                out["aux"] = torch.zeros(0, len(CFG.ALL_TARGET_COLS), device=device, dtype=dtype)
            if return_features:
                out["feature_maps"] = {}
            return out

        def forward(self, *inputs, x_left=None, x_right=None, return_features: bool = False):
            if inputs:
                if len(inputs) == 1:
                    first = inputs[0]
                    if isinstance(first, (tuple, list)):
                        if len(first) >= 1:
                            x_left = first[0]
                        if len(first) >= 2:
                            x_right = first[1]
                    else:
                        x_left = first
                else:
                    x_left = inputs[0]
                    x_right = inputs[1]

            if x_left is None:
                return self._empty_forward_output(return_features=return_features)
            if isinstance(x_left, torch.Tensor) and x_left.shape[0] == 0:
                return self._empty_forward_output(return_features=return_features)

            if x_right is None:
                if isinstance(x_left, torch.Tensor):
                    if x_left.shape[1] % 2 != 0:
                        raise ValueError("Cannot infer left/right inputs from a single tensor.")
                    x_left, x_right = torch.chunk(x_left, 2, dim=1)
                else:
                    raise ValueError("Missing x_right input.")

            feat_l, feats_l = self._half_forward(x_left)
            feat_r, feats_r = self._half_forward(x_right)
            total, gdm, green, f_concat = self._merge_heads(feat_l, feat_r)

            out = {
                "total": total,
                "gdm": gdm,
                "green": green,
                "score_feat": f_concat,
            }
            pred = torch.cat([green, total - gdm, gdm - green, gdm, total], dim=1)
            out["pred"] = pred
            if self.aux_head is not None:
                aux_tokens = torch.cat([feats_l["stage2_tokens"], feats_r["stage2_tokens"]], dim=1)
                aux_pred = self.softplus(self.aux_head(aux_tokens.mean(dim=1)))
                out["aux"] = aux_pred
            if return_features:
                out["feature_maps"] = {
                    "stage1_left": feats_l.get("stage1_map"),
                    "stage1_right": feats_r.get("stage1_map"),
                    "stage3_left": feats_l.get("stage3_tokens"),
                    "stage3_right": feats_r.get("stage3_tokens"),
                }
            return out

    class SimpleViTRegressor(nn.Module):
        """Single ViT backbone regressor for simplified experiments."""

        def __init__(self, backbone: str = "vit_small_patch14_dinov2", img_size: int = 384, dropout: float = 0.1):
            super().__init__()
            self.backbone_name = backbone
            self.input_res = img_size
            self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
            self.head = nn.Sequential(
                nn.LayerNorm(self.backbone.num_features),
                nn.Dropout(dropout),
                nn.Linear(self.backbone.num_features, len(CFG.ALL_TARGET_COLS)),
            )

        def summary(self) -> str:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            lines = [
                f"Model: {self.__class__.__name__}",
                f"Backbone: {self.backbone_name}",
                f"Input resolution: {self.input_res}",
                f"Total parameters: {total_params:,}",
                f"Trainable parameters: {trainable_params:,}",
                "",
                "TrainCFG:",
            ]
            cfg_dict = asdict(CFG)
            for key in sorted(cfg_dict.keys()):
                lines.append(f"  {key}: {cfg_dict[key]}")
            return "\n".join(lines)

        def forward(self, *inputs, x=None, x_left=None, x_right=None, **kwargs):
            if x is None:
                if x_left is None:
                    raise ValueError("SimpleViTRegressor expects x or x_left input.")
                if x_right is not None:
                    x = torch.cat([x_left, x_right], dim=3)
                else:
                    x = x_left
            feat = self.backbone(x)
            pred = self.head(feat)
            out = {
                "pred": pred,
                "green": pred[:, 0:1],
                "gdm": pred[:, 3:4],
                "total": pred[:, 4:5],
            }
            return out

else:
    class CrossPVT_T2T_MambaDINO:
        """Fallback stub when torch/timm are unavailable."""

        def __init__(self, *args, **kwargs):
            missing = []
            if torch is None:
                missing.append("torch")
            if timm is None:
                missing.append("timm")
            missing_str = ", ".join(missing) if missing else "torch and timm"
            raise ImportError(f"CrossPVT_T2T_MambaDINO requires {missing_str} to be installed.")
    class SimpleViTRegressor:
        def __init__(self, *args, **kwargs):
            missing = []
            if torch is None:
                missing.append("torch")
            if timm is None:
                missing.append("timm")
            missing_str = ", ".join(missing) if missing else "torch and timm"
            raise ImportError(f"SimpleViTRegressor requires {missing_str} to be installed.")


__all__ = [
    "ModelConfig",
    "TrainCFG",
    "CFG",
    "create_model",
    "build_cv",
    "rmse",
    "update_cfg_from_checkpoint",
    "CrossPVT_T2T_MambaDINO",
    "SimpleViTRegressor",
]
