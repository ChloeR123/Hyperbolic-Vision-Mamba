import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import Tensor
from typing import Optional

import geoopt
from geoopt import ManifoldParameter
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from collections import namedtuple

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

from rope import *
import random

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

__all__ = [
    'vim_tiny_patch16_224', 'vim_small_patch16_224', 'vim_base_patch16_224',
    'vim_tiny_patch16_384', 'vim_small_patch16_384', 'vim_base_patch16_384',
]




class StablePoincareBall(geoopt.PoincareBall):
    def __init__(self, c=0.5):
        super().__init__(c=c)

    def expmap0(self, u: torch.Tensor) -> torch.Tensor:
        """
        原点处的指数映射：将欧氏切空间向量u映射到双曲空间
        对应《双曲图像嵌入》论文公式8（原点处简化）
        Args:
            u: 欧氏切空间向量，形状为 [*, D]（*为任意批次维度，D为特征维度）
        Returns:
            双曲空间中的点，形状为 [*, D]
        """
        # 1. 数值稳定性：限制向量范数，避免除零和溢出
        u_norm = u.norm(dim=-1, keepdim=True).clamp(min=1e-8, max=5.0)  # 最小1e-8防除零，最大5.0防tanh溢出
        u_safe = u / u_norm  # 单位方向向量

        # 2. 计算论文公式中的核心项
        sqrt_c = math.sqrt(self.c)
        norm_scaled = sqrt_c * u_norm  # √c · ||u||
        tanh_term = torch.tanh(norm_scaled)  # tanh(√c · ||u||)

        # 3. 计算最终映射结果
        exp_result = (tanh_term / sqrt_c) * u_safe

        # 4. 确保结果在双曲空间内（范数 < 1/√c），留出5%安全边界
        max_allowed_norm = 0.95 / sqrt_c
        exp_norm = exp_result.norm(dim=-1, keepdim=True)
        scale = torch.where(exp_norm > max_allowed_norm, max_allowed_norm / exp_norm, torch.ones_like(exp_norm))
        exp_result = exp_result * scale

        return exp_result

    def logmap0(self, y: torch.Tensor) -> torch.Tensor:
        """
        原点处的对数映射：将双曲空间点y映射回欧氏切空间
        对应《双曲图像嵌入》论文公式9（原点处简化）
        Args:
            y: 双曲空间中的点，形状为 [*, D]
        Returns:
            欧氏切空间向量，形状为 [*, D]
        """
        # 1. 数值稳定性：限制点的范数，避免接近双曲空间边界
        sqrt_c = math.sqrt(self.c)
        max_allowed_norm = 0.95 / sqrt_c  # 安全边界（95% of 1/√c）
        y_norm = y.norm(dim=-1, keepdim=True).clamp(min=1e-8, max=max_allowed_norm)
        y_safe = y / y_norm  # 单位方向向量

        # 2. 计算论文公式中的核心项
        norm_scaled = sqrt_c * y_norm  # √c · ||y||
        # arctanh限制输入<0.95，避免arctanh(1)→+inf
        arctanh_term = torch.atanh(norm_scaled.clamp(max=0.95))

        # 3. 计算最终映射结果
        log_result = (arctanh_term / sqrt_c) * y_safe

        return log_result

GLOBAL_MANIFOLD = StablePoincareBall(c=0.5)

class HyperbolicLogisticRegression(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.manifold = GLOBAL_MANIFOLD  # 关联全局双曲空间
        self.num_classes = num_classes  # 类别数
        self.in_features = in_features  # 输入特征维度

        # 可学习参数（对应论文中的p_y和w_y）
        self.p_y = ManifoldParameter(
            torch.zeros(num_classes, in_features),
            manifold=self.manifold  # p_y：每个类别的双曲偏移点
        )
        self.w_y = nn.Parameter(
            torch.zeros(num_classes, in_features)  # w_y：每个类别的方向向量（切空间）
        )

        # 参数初始化（小范围初始化，避免接近双曲边界）
        nn.init.normal_(self.p_y, std=0.01)
        nn.init.normal_(self.w_y, std=0.01)

    def forward(self, x):
        """
        计算双曲逻辑回归的Logit（对应《双曲图像语义分割》论文公式7）
        Args:
            x: 双曲空间中的像素/图像特征，形状为 [B, D]（B=批次大小，D=特征维度）
        Returns:
            logits: 每个类别的Logit值，形状为 [B, num_classes]
        """
        B, D = x.shape
        C = self.num_classes

        # 扩展维度以支持广播计算（匹配批次和类别）
        x_expanded = x.unsqueeze(1)  # [B, 1, D]
        p_y_expanded = self.p_y.unsqueeze(0)  # [1, C, D]
        w_y_expanded = self.w_y.unsqueeze(0)  # [1, C, D]

        # -------------------------- 修复部分 --------------------------
        # 错误原因：geoopt的mobius_scalar_mul要求标量r是Tensor，而非float
        # 修复方案：将-1.0转为与p_y_expanded同设备、同dtype的Tensor
        device = p_y_expanded.device  # 获取参数设备（CPU/GPU）
        dtype = p_y_expanded.dtype  # 获取参数数据类型（如float32）
        r = torch.tensor(-1.0, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, 1]（适配广播）

        # 步骤1：计算 -p_y（双曲空间标量乘法：r=-1.0 · p_y）
        neg_p_y = self.manifold.mobius_scalar_mul(r, p_y_expanded)  # [1, C, D]（类型匹配）
        # -------------------------------------------------------------

        # 步骤2：计算 Möbius 加法：-p_y ⊕_c x（双曲空间加法，符合《双曲图像嵌入》公式3）
        mobius_add = self.manifold.mobius_add(neg_p_y, x_expanded)  # [B, C, D]

        # 步骤3：计算内积 ⟨-p_y ⊕_c x, w_y⟩（符合《双曲图像语义分割》公式7分子项）
        inner_product = (mobius_add * w_y_expanded).sum(dim=-1, keepdim=True)  # [B, C, 1]

        # 步骤4：计算 ||-p_y ⊕_c x||²（Möbius加法结果的平方范数，公式7分母项）
        mobius_norm_sq = mobius_add.norm(dim=-1, keepdim=True).pow(2)  # [B, C, 1]

        # 步骤5：计算 ||w_y||（方向向量范数，防除零，符合公式7）
        w_norm = w_y_expanded.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [1, C, 1]

        # 步骤6：计算共形因子 λ_py^c = 2 / (1 - c·||p_y||²)（《双曲图像嵌入》公式1的 conformal factor）
        p_y_norm_sq = p_y_expanded.norm(dim=-1, keepdim=True).pow(2)  # [1, C, 1]
        lambda_p = 2.0 / (1.0 - self.manifold.c * p_y_norm_sq)  # [1, C, 1]

        # 步骤7：计算论文公式中的arcsinh项（《双曲图像语义分割》公式7核心）
        sqrt_c = math.sqrt(self.manifold.c)
        numerator = 2.0 * sqrt_c * inner_product  # 分子：2√c · ⟨-p_y⊕x, w_y⟩
        denominator = (1.0 - self.manifold.c * mobius_norm_sq) * w_norm  # 分母：(1 - c·||-p_y⊕x||²)·||w_y||
        denominator = denominator.clamp(min=1e-8)  # 防除零

        # 步骤8：计算arcsinh（限制输入范围，防溢出）
        arcsinh_arg = (numerator / denominator).clamp(min=-1e8, max=1e8)
        arcsinh_term = torch.arcsinh(arcsinh_arg)  # [B, C, 1]

        # 步骤9：计算最终Logit（公式7完整结果）
        logits = (lambda_p * w_norm / sqrt_c) * arcsinh_term  # [B, C, 1]

        # 去除多余维度，返回 [B, C]
        return logits.squeeze(-1)


class HyperbolicDistanceHead(nn.Module):
    """双曲测地线距离分类头"""

    def __init__(self, in_features, num_classes):
        super().__init__()
        self.manifold = GLOBAL_MANIFOLD
        self.num_classes = num_classes

        # 新初始化方式：从切空间映射，确保稀疏分布
        init_tangent = torch.randn(num_classes, in_features) * 0.1  # 增大标准差
        self.prototypes = ManifoldParameter(self.manifold.expmap0(init_tangent), manifold=GLOBAL_MANIFOLD)
        # 强制约束在双曲空间内（范数 < 1/sqrt(c)）
        with torch.no_grad():
            max_norm = 1.0 / torch.sqrt(torch.tensor(GLOBAL_MANIFOLD.c)) - 1e-5
            norm = self.prototypes.norm(dim=-1, keepdim=True)
            self.prototypes.data = self.prototypes.data * (max_norm / norm.clamp(min=1e-8))

    def forward(self, x):
        """
        计算输入与类别原型的负双曲测地线距离
        Args:
            x: 输入特征 [B, D] (双曲空间)
        Returns:
            logits: 负测地线距离 [B, num_classes]
        """
        # 扩展维度用于广播计算
        x_hyp = x.unsqueeze(1)  # [B, 1, D]
        prototypes = self.prototypes.unsqueeze(0)  # [1, C, D]

        # 计算测地线距离 (使用PoincareBall的dist方法)
        distances = self.manifold.dist(x_hyp, prototypes)  # [B, C]
        logits = -distances

        return logits


# 双曲层归一化
class HyperbolicLayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-5):
        super().__init__()
        self.manifold = GLOBAL_MANIFOLD
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        # 映射到原点切空间
        x_tangent = self.manifold.logmap0(x)
        # 应用层归一化
        mean = x_tangent.mean(dim=-1, keepdim=True)
        std = x_tangent.std(dim=-1, keepdim=True)
        x_norm = (x_tangent - mean) / (std + self.eps)
        x_norm = x_norm * self.gamma + self.beta
        # 映射回双曲空间
        return self.manifold.expmap0(x_norm)


# 双曲线性层
class HyperbolicLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.manifold = GLOBAL_MANIFOLD
        self.in_features = in_features
        self.out_features = out_features

        self.weight = ManifoldParameter(
            torch.Tensor(out_features, in_features), manifold=GLOBAL_MANIFOLD
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = ManifoldParameter(
                torch.zeros(out_features), manifold=GLOBAL_MANIFOLD
            )
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, input):
        # 双曲矩阵乘法
        output = self.manifold.mobius_matvec(self.weight, input)
        if self.bias is not None:
            # 双曲加法
            output = self.manifold.mobius_add(output, self.bias)
        return output


# 双曲位置编码
class HyperbolicPosEmbed(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.manifold = GLOBAL_MANIFOLD
        self.pos_embed = ManifoldParameter(
            torch.zeros(1, num_patches, embed_dim), manifold=GLOBAL_MANIFOLD
        )
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        # 双曲加法
        return self.manifold.mobius_add(x, self.pos_embed)


# 双曲Patch Embedding
class HyperbolicPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, flatten=True,clip_radius=1.0):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.manifold = GLOBAL_MANIFOLD

        # 添加裁剪半径参数
        self.clip_radius = clip_radius
        self.clipping_applied = False  # 确保只裁剪一次

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.flatten_op = nn.Flatten(2) if flatten else nn.Identity()

        self.hyperbolic_proj = HyperbolicLinear(embed_dim, embed_dim)
        self.norm = HyperbolicLayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # 欧几里得空间投影
        x = self.flatten_op(x)
        if isinstance(self.flatten_op, nn.Flatten):
            x = x.transpose(1, 2)

        # ==================== 关键修改：添加裁剪操作 ====================
        # 只在第一次前向传播时应用裁剪（训练和推理都适用）
        # if not self.clipping_applied:
            # r = self.clip_radius  # 使用预设的裁剪半径
            # norm = torch.norm(x, p=2, dim=-1, keepdim=True)
            # 防止除零
            # norm = torch.clamp(norm, min=1e-8)
            # 裁剪条件：范数大于r时缩放至r
            # scale = torch.where(norm > r, r / norm, torch.ones_like(norm))
            # x = x * scale
            # self.clipping_applied = True  # 标记已应用裁剪
        # ==============================================================

        # # 添加特征裁剪（在指数映射前）
        # r = 0.8  # 裁剪半径，根据论文建议设为1.0
        # norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        # # 防止除零
        # norm = torch.clamp(norm, min=1e-8)
        # # 裁剪条件：范数大于r时缩放至r
        # scale = torch.where(norm > r, r / norm, torch.ones_like(norm))
        # x = x * scale

        x = self.manifold.expmap0(x)  # 映射到双曲空间
        x = self.hyperbolic_proj(x)  # 双曲空间投影
        x = self.norm(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.manifold = GLOBAL_MANIFOLD
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                # 双曲空间残差连接
                residual = self.manifold.mobius_add(residual, self.drop_path(hidden_states))

            # 双曲层归一化
            hidden_states = self.norm(residual)
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            # 不支持融合操作
            raise NotImplementedError("Fused add-norm not supported in hyperbolic space")

        # Mamba操作在切空间进行
        hidden_states_tangent = self.manifold.logmap0(hidden_states)
        hidden_states_tangent = self.mixer(hidden_states_tangent, inference_params=inference_params)
        hidden_states = self.manifold.expmap0(hidden_states_tangent)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        d_state=16,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        if_bimamba=False,
        bimamba_type="none",
        if_divide_out=False,
        init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, d_state=d_state, layer_idx=layer_idx, bimamba_type=bimamba_type,
                        if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)

    # 使用双曲层归一化
    norm_cls = partial(
        HyperbolicLayerNorm, eps=norm_epsilon
    )

    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        # 增大初始化标准差
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.normal_(m.bias, std=0.005)  # 偏置也使用正态初始化
    elif isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)  # 增强卷积层初始化
        if m.bias is not None:
            nn.init.normal_(m.bias, std=0.005)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    # 双曲参数特殊初始化
    elif isinstance(m, ManifoldParameter) and hasattr(m, 'manifold'):
        with torch.no_grad():
            # 生成初始化点（切空间）
            init_tangent = torch.randn_like(m.data) * 0.01  # 更小的噪声
            # 映射到双曲空间
            m.data = m.manifold.expmap0(init_tangent)
            # 范数裁剪：确保点不靠近边界
            max_norm = 0.9 / torch.sqrt(torch.tensor(m.manifold.c))  # 90% 边界
            norm = m.data.norm(dim=-1, keepdim=True)
            # 缩放范数过大的点
            scale = torch.clamp(max_norm / norm, max=1.0)
            m.data = m.data * scale


class VisionMamba(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 stride=16,
                 depth=12,
                 embed_dim=256,
                 d_state=16,
                 channels=3,
                 num_classes=1000,
                 ssm_cfg=None,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = True,
                 initializer_cfg=None,
                 fused_add_norm=True,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 if_bidirectional=False,
                 final_pool_type='none',
                 if_abs_pos_embed=True,
                 if_rope=False,
                 if_rope_residual=False,
                 flip_img_sequences_ratio=-1.,
                 if_bimamba=False,
                 bimamba_type="v2",
                 if_cls_token=True,
                 if_divide_out=True,
                 init_layer_scale=None,
                 use_double_cls_token=False,
                 use_middle_cls_token=True,
                 clip_radius=1.0,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0
        self.manifold = GLOBAL_MANIFOLD

        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim

        # 使用双曲Patch Embedding
        self.patch_embed = HyperbolicPatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride,
            in_chans=channels, embed_dim=embed_dim, clip_radius=clip_radius  # 传递裁剪参数
        )
        num_patches = self.patch_embed.num_patches

        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = ManifoldParameter(torch.zeros(1, 1, embed_dim),
                                                        manifold=self.manifold)
                self.cls_token_tail = ManifoldParameter(torch.zeros(1, 1, embed_dim),
                                                        manifold=self.manifold)
                self.num_tokens = 2
            else:
                self.cls_token = ManifoldParameter(torch.zeros(1, 1, embed_dim),
                                                   manifold=self.manifold)

        if if_abs_pos_embed:
            self.pos_embed = HyperbolicPosEmbed(num_patches + self.num_tokens, embed_dim)
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len
            )

        # 使用双曲测地线距离分类头
        if num_classes > 0:
            self.head = HyperbolicLogisticRegression(embed_dim, num_classes)
        else:
            self.head = nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    d_state=d_state,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_divide_out=if_divide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        # 双曲归一化
        self.norm_f = HyperbolicLayerNorm(embed_dim, eps=norm_epsilon)

        self.patch_embed.apply(segm_init_weights)

        if not isinstance(self.head, HyperbolicDistanceHead):
            self.head.apply(segm_init_weights)

        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)

        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None, if_random_cls_token_position=False,
                         if_random_token_rank=False):
        x = self.patch_embed(x)
        B, M, _ = x.shape

        if self.if_cls_token:
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, M + 1]
                # 双曲空间连接
                x = torch.cat((
                    cls_token_head,
                    x,
                    cls_token_tail
                ), dim=1)
            else:
                if self.use_middle_cls_token:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = M // 2
                    x = torch.cat((
                        x[:, :token_position, :],
                        cls_token,
                        x[:, token_position:, :]
                    ), dim=1)
                elif if_random_cls_token_position:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = random.randint(0, M)
                    x = torch.cat((
                        x[:, :token_position, :],
                        cls_token,
                        x[:, token_position:, :]
                    ), dim=1)
                else:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = 0
                    x = torch.cat((cls_token, x), dim=1)
            M = x.shape[1]

        if self.if_abs_pos_embed:
            x = self.pos_embed(x)
            x = self.pos_drop(x)

        if if_random_token_rank:
            shuffle_indices = torch.randperm(M)
            x = x[:, shuffle_indices, :]
            if isinstance(token_position, list):
                token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in
                                  range(len(token_position))]
            else:
                token_position = torch.where(shuffle_indices == token_position)[0].item()

        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip_img_sequences = True

        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:
                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
        else:
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1]),
                    inference_params=inference_params
                )
                # 双曲空间加法
                hidden_states = self.manifold.mobius_add(hidden_states_f, hidden_states_b.flip([1]))
                residual = self.manifold.mobius_add(residual_f, residual_b.flip([1]))

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                # 双曲空间残差连接
                residual = self.manifold.mobius_add(residual, self.drop_path(hidden_states))
            hidden_states = self.norm_f(residual)
        else:
            raise NotImplementedError

        if self.if_cls_token:
            if self.use_double_cls_token:
                return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
            else:
                return hidden_states[:, token_position, :]

        if self.final_pool_type == 'none':
            return hidden_states[:, -1, :]
        elif self.final_pool_type == 'mean':
            # 双曲平均
            return self.manifold.karcher_mean(hidden_states, dim=1)
        elif self.final_pool_type == 'max':
            return hidden_states
        elif self.final_pool_type == 'all':
            return hidden_states
        else:
            raise NotImplementedError

    def forward(self, x, return_features=False, inference_params=None, if_random_cls_token_position=False,
                if_random_token_rank=False):
        x = self.forward_features(x, inference_params, if_random_cls_token_position=if_random_cls_token_position,
                                  if_random_token_rank=if_random_token_rank)
        if return_features:
            return x
        x = self.head(x)
        if self.final_pool_type == 'max':
            x = x.max(dim=1)[0]
        return x


# 注册双曲Mamba模型
@register_model
def hyper_vim_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=256, depth=24, rms_norm=False, residual_in_fp32=False, fused_add_norm=False,
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False,
        bimamba_type="v2", if_cls_token=True, use_middle_cls_token=True,manifold=GLOBAL_MANIFOLD, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def hyper_vim_small_patch16_224(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=384, depth=24, rms_norm=False,
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False,
        bimamba_type="v2", if_cls_token=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def hyper_vim_base_patch16_224(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=768, depth=24, rms_norm=False,
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False,
        bimamba_type="v2", if_cls_token=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    return model


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from functools import partial
# from torch import Tensor
# from typing import Optional
#
# import geoopt
# from geoopt import ManifoldParameter
# from timm.models.vision_transformer import VisionTransformer, _cfg
# from timm.models.registry import register_model
# from timm.models.layers import trunc_normal_, lecun_normal_
#
# from timm.models.layers import DropPath, to_2tuple
# from timm.models.vision_transformer import _load_weights
#
# import math
#
# from collections import namedtuple
#
# from mamba_ssm.modules.mamba_simple import Mamba
# from mamba_ssm.utils.generation import GenerationMixin
# from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
#
# from rope import *
# import random
#
# try:
#     from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
# except ImportError:
#     RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
#
# __all__ = [
#     'vim_tiny_patch16_224', 'vim_small_patch16_224', 'vim_base_patch16_224',
#     'vim_tiny_patch16_384', 'vim_small_patch16_384', 'vim_base_patch16_384',
# ]
#
# GLOBAL_MANIFOLD = geoopt.PoincareBall(c=0.5)
#
# class StablePoincareBall(geoopt.PoincareBall):
#     def __init__(self, c=0.5):
#         super().__init__(c=c)
#     def expmap0(self, x, u):
#         # 添加梯度裁剪防止NaN
#         u_norm = u.norm(dim=-1, keepdim=True).clamp(min=1e-8, max=5.0)
#         return super().expmap0(x, u / u_norm * u_norm.clamp(max=self.radius*0.9))
#
#     def logmap0(self, y: torch.Tensor) -> torch.Tensor:
#         # 添加安全截断
#         y_norm = y.norm(dim=-1, keepdim=True).clamp(max=0.95 * (1 / torch.sqrt(self.c)))
#         return super().logmap0(y / y_norm * y_norm)  # 确保在安全范围内
#
# class HyperbolicLogisticRegression(nn.Module):
#     def __init__(self, in_features, num_classes):
#         super().__init__()
#         self.manifold = GLOBAL_MANIFOLD
#         self.num_classes = num_classes
#         # 增加中间层增强梯度传播
#         self.proj = nn.Linear(in_features, in_features)  # 欧氏空间投影
#         self.weight = nn.Parameter(torch.zeros(num_classes, in_features))
#         self.bias = nn.Parameter(torch.zeros(num_classes))
#         nn.init.kaiming_normal_(self.proj.weight, nonlinearity='relu')
#         nn.init.normal_(self.weight, std=0.01)
#         nn.init.zeros_(self.bias)
#         self.grad_scale = nn.Parameter(torch.tensor(1.0))  # 梯度缩放因子
#
#     def forward(self, x):
#         x_tangent = self.manifold.logmap0(x)
#         x_proj = F.relu(self.proj(x_tangent))  # 增加非线性变换
#         # 缩放梯度流
#         x_proj = x_proj * self.grad_scale + x_tangent  # 残差连接
#         logits = F.linear(x_proj, self.weight, self.bias)
#         return logits
#
# class HyperbolicDistanceHead(nn.Module):
#     """双曲测地线距离分类头"""
#
#     def __init__(self, in_features, num_classes):
#         super().__init__()
#         self.manifold = GLOBAL_MANIFOLD
#         self.num_classes = num_classes
#
#         # 新初始化方式：从切空间映射，确保稀疏分布
#         init_tangent = torch.randn(num_classes, in_features) * 0.1  # 增大标准差
#         self.prototypes = ManifoldParameter(GLOBAL_MANIFOLD.expmap0(init_tangent), manifold=GLOBAL_MANIFOLD)
#         # 强制约束在双曲空间内（范数 < 1/sqrt(c)）
#         with torch.no_grad():
#             max_norm = 1.0 / torch.sqrt(torch.tensor(GLOBAL_MANIFOLD.c)) - 1e-5
#             norm = self.prototypes.norm(dim=-1, keepdim=True)
#             self.prototypes.data = self.prototypes.data * (max_norm / norm.clamp(min=1e-8))
#
#     def forward(self, x):
#         """
#         计算输入与类别原型的负双曲测地线距离
#         Args:
#             x: 输入特征 [B, D] (双曲空间)
#         Returns:
#             logits: 负测地线距离 [B, num_classes]
#         """
#         # 扩展维度用于广播计算
#         x_hyp = x.unsqueeze(1)  # [B, 1, D]
#         prototypes = self.prototypes.unsqueeze(0)  # [1, C, D]
#
#         # 计算测地线距离 (使用PoincareBall的dist方法)
#         distances = self.manifold.dist(x_hyp, prototypes)  # [B, C]
#         logits = -distances
#
#         return logits
#
# # 双曲层归一化
# class HyperbolicLayerNorm(nn.Module):
#     def __init__(self, embedding_dim, eps=1e-5):
#         super().__init__()
#         self.manifold = GLOBAL_MANIFOLD
#         self.eps = eps
#         self.gamma = nn.Parameter(torch.ones(embedding_dim))
#         self.beta = nn.Parameter(torch.zeros(embedding_dim))
#
#     def forward(self, x):
#         # 映射到原点切空间
#         x_tangent = self.manifold.logmap0(x)
#         # 应用层归一化
#         mean = x_tangent.mean(dim=-1, keepdim=True)
#         std = x_tangent.std(dim=-1, keepdim=True)
#         x_norm = (x_tangent - mean) / (std + self.eps)
#         x_norm = x_norm * self.gamma + self.beta
#         # 映射回双曲空间
#         return self.manifold.expmap0(x_norm)
#
# # 双曲线性层
# class HyperbolicLinear(nn.Module):
#     def __init__(self, in_features, out_features, bias=True):
#         super().__init__()
#         self.manifold = GLOBAL_MANIFOLD
#         self.in_features = in_features
#         self.out_features = out_features
#
#         self.weight = ManifoldParameter(
#             torch.Tensor(out_features, in_features), manifold=GLOBAL_MANIFOLD
#         )
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if bias:
#             self.bias = ManifoldParameter(
#                 torch.zeros(out_features), manifold=GLOBAL_MANIFOLD
#             )
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.bias, -bound, bound)
#         else:
#             self.bias = None
#
#     def forward(self, input):
#         # 双曲矩阵乘法
#         output = self.manifold.mobius_matvec(self.weight, input)
#         if self.bias is not None:
#             # 双曲加法
#             output = self.manifold.mobius_add(output, self.bias)
#         return output
#
# # 双曲位置编码
# class HyperbolicPosEmbed(nn.Module):
#     def __init__(self, num_patches, embed_dim):
#         super().__init__()
#         self.manifold = GLOBAL_MANIFOLD
#         self.pos_embed = ManifoldParameter(
#             torch.zeros(1, num_patches, embed_dim), manifold=GLOBAL_MANIFOLD
#         )
#         trunc_normal_(self.pos_embed, std=.02)
#
#     def forward(self, x):
#         # 双曲加法
#         return self.manifold.mobius_add(x, self.pos_embed)
#
# # 双曲Patch Embedding
# class HyperbolicPatchEmbed(nn.Module):
#     def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, flatten=True):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#         self.flatten = flatten
#         self.manifold = GLOBAL_MANIFOLD
#
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
#         self.flatten_op = nn.Flatten(2) if flatten else nn.Identity()
#
#         self.hyperbolic_proj = HyperbolicLinear(embed_dim, embed_dim)
#         self.norm = HyperbolicLayerNorm(embed_dim)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.proj(x)  # 欧几里得空间投影
#         x = self.flatten_op(x)
#         if isinstance(self.flatten_op, nn.Flatten):
#             x = x.transpose(1, 2)
#
#         # 添加特征裁剪（在指数映射前）
#         r = 0.8  # 裁剪半径，根据论文建议设为1.0
#         norm = torch.norm(x, p=2, dim=-1, keepdim=True)
#         # 防止除零
#         norm = torch.clamp(norm, min=1e-8)
#         # 裁剪条件：范数大于r时缩放至r
#         scale = torch.where(norm > r, r / norm, torch.ones_like(norm))
#         x = x * scale
#
#         x = self.manifold.expmap0(x)  # 映射到双曲空间
#         x = self.hyperbolic_proj(x)  # 双曲空间投影
#         x = self.norm(x)
#         return x
#
# class Block(nn.Module):
#     def __init__(
#             self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.):
#         super().__init__()
#         self.residual_in_fp32 = residual_in_fp32
#         self.fused_add_norm = fused_add_norm
#         self.manifold = GLOBAL_MANIFOLD
#         self.mixer = mixer_cls(dim)
#         self.norm = norm_cls(dim)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         if self.fused_add_norm:
#             assert RMSNorm is not None, "RMSNorm import fails"
#             assert isinstance(
#                 self.norm, (nn.LayerNorm, RMSNorm)
#             ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
#
#     def forward(
#             self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
#     ):
#         if not self.fused_add_norm:
#             if residual is None:
#                 residual = hidden_states
#             else:
#                 # 双曲空间残差连接
#                 residual = self.manifold.mobius_add(residual, self.drop_path(hidden_states))
#
#             # 双曲层归一化
#             hidden_states = self.norm(residual)
#             if self.residual_in_fp32:
#                 residual = residual.to(torch.float32)
#         else:
#             # 不支持融合操作
#             raise NotImplementedError("Fused add-norm not supported in hyperbolic space")
#
#         # Mamba操作在切空间进行
#         hidden_states_tangent = self.manifold.logmap0(hidden_states)
#         hidden_states_tangent = self.mixer(hidden_states_tangent, inference_params=inference_params)
#         hidden_states = self.manifold.expmap0(hidden_states_tangent)
#
#         return hidden_states, residual
#
#     def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
#         return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
#
# def create_block(
#         d_model,
#         d_state=16,
#         ssm_cfg=None,
#         norm_epsilon=1e-5,
#         drop_path=0.,
#         rms_norm=False,
#         residual_in_fp32=False,
#         fused_add_norm=False,
#         layer_idx=None,
#         device=None,
#         dtype=None,
#         if_bimamba=False,
#         bimamba_type="none",
#         if_divide_out=False,
#         init_layer_scale=None,
# ):
#     if if_bimamba:
#         bimamba_type = "v1"
#     if ssm_cfg is None:
#         ssm_cfg = {}
#     factory_kwargs = {"device": device, "dtype": dtype}
#     mixer_cls = partial(Mamba, d_state=d_state, layer_idx=layer_idx, bimamba_type=bimamba_type,
#                         if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
#
#     # 使用双曲层归一化
#     norm_cls = partial(
#         HyperbolicLayerNorm, eps=norm_epsilon
#     )
#
#     block = Block(
#         d_model,
#         mixer_cls,
#         norm_cls=norm_cls,
#         drop_path=drop_path,
#         fused_add_norm=fused_add_norm,
#         residual_in_fp32=residual_in_fp32,
#     )
#     block.layer_idx = layer_idx
#     return block
#
# def _init_weights(
#         module,
#         n_layer,
#         initializer_range=0.02,
#         rescale_prenorm_residual=True,
#         n_residuals_per_layer=1,
# ):
#     if isinstance(module, nn.Linear):
#         if module.bias is not None:
#             if not getattr(module.bias, "_no_reinit", False):
#                 nn.init.zeros_(module.bias)
#     elif isinstance(module, nn.Embedding):
#         nn.init.normal_(module.weight, std=initializer_range)
#
#     if rescale_prenorm_residual:
#         for name, p in module.named_parameters():
#             if name in ["out_proj.weight", "fc2.weight"]:
#                 nn.init.kaiming_uniform_(p, a=math.sqrt(5))
#                 with torch.no_grad():
#                     p /= math.sqrt(n_residuals_per_layer * n_layer)
#
# def segm_init_weights(m):
#     if isinstance(m, nn.Linear):
#         # 增大初始化标准差
#         trunc_normal_(m.weight, std=0.02)
#         if isinstance(m, nn.Linear) and m.bias is not None:
#             nn.init.normal_(m.bias, std=0.005)  # 偏置也使用正态初始化
#     elif isinstance(m, nn.Conv2d):
#         lecun_normal_(m.weight)  # 增强卷积层初始化
#         if m.bias is not None:
#             nn.init.normal_(m.bias, std=0.005)
#     elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
#         nn.init.zeros_(m.bias)
#         nn.init.ones_(m.weight)
#     # 双曲参数特殊初始化
#     elif isinstance(m, ManifoldParameter) and hasattr(m, 'manifold'):
#         with torch.no_grad():
#             # 生成初始化点（切空间）
#             init_tangent = torch.randn_like(m.data) * 0.01  # 更小的噪声
#             # 映射到双曲空间
#             m.data = m.manifold.expmap0(init_tangent)
#             # 范数裁剪：确保点不靠近边界
#             max_norm = 0.9 / torch.sqrt(torch.tensor(m.manifold.c))  # 90% 边界
#             norm = m.data.norm(dim=-1, keepdim=True)
#             # 缩放范数过大的点
#             scale = torch.clamp(max_norm / norm, max=1.0)
#             m.data = m.data * scale
#
# class VisionMamba(nn.Module):
#     def __init__(self,
#                  img_size=224,
#                  patch_size=16,
#                  stride=16,
#                  depth=24,
#                  embed_dim=192,
#                  d_state=16,
#                  channels=3,
#                  num_classes=1000,
#                  ssm_cfg=None,
#                  drop_rate=0.,
#                  drop_path_rate=0.1,
#                  norm_epsilon: float = 1e-5,
#                  rms_norm: bool = True,
#                  initializer_cfg=None,
#                  fused_add_norm=True,
#                  residual_in_fp32=True,
#                  device=None,
#                  dtype=None,
#                  ft_seq_len=None,
#                  pt_hw_seq_len=14,
#                  if_bidirectional=False,
#                  final_pool_type='none',
#                  if_abs_pos_embed=True,
#                  if_rope=False,
#                  if_rope_residual=False,
#                  flip_img_sequences_ratio=-1.,
#                  if_bimamba=False,
#                  bimamba_type="v2",
#                  if_cls_token=True,
#                  if_divide_out=True,
#                  init_layer_scale=None,
#                  use_double_cls_token=False,
#                  use_middle_cls_token=True,
#                  **kwargs):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         kwargs.update(factory_kwargs)
#         super().__init__()
#         self.residual_in_fp32 = residual_in_fp32
#         self.fused_add_norm = fused_add_norm
#         self.if_bidirectional = if_bidirectional
#         self.final_pool_type = final_pool_type
#         self.if_abs_pos_embed = if_abs_pos_embed
#         self.if_rope = if_rope
#         self.if_rope_residual = if_rope_residual
#         self.flip_img_sequences_ratio = flip_img_sequences_ratio
#         self.if_cls_token = if_cls_token
#         self.use_double_cls_token = use_double_cls_token
#         self.use_middle_cls_token = use_middle_cls_token
#         self.num_tokens = 1 if if_cls_token else 0
#         self.manifold = GLOBAL_MANIFOLD
#
#         self.num_classes = num_classes
#         self.d_model = self.num_features = self.embed_dim = embed_dim
#
#         # 使用双曲Patch Embedding
#         self.patch_embed = HyperbolicPatchEmbed(
#             img_size=img_size, patch_size=patch_size, stride=stride,
#             in_chans=channels, embed_dim=embed_dim
#         )
#         num_patches = self.patch_embed.num_patches
#
#         if if_cls_token:
#             if use_double_cls_token:
#                 self.cls_token_head = ManifoldParameter(torch.zeros(1, 1, embed_dim),
#                                                         manifold=self.manifold)
#                 self.cls_token_tail = ManifoldParameter(torch.zeros(1, 1, embed_dim),
#                                                         manifold=self.manifold)
#                 self.num_tokens = 2
#             else:
#                 self.cls_token = ManifoldParameter(torch.zeros(1, 1, embed_dim),
#                                                    manifold=self.manifold)
#
#         if if_abs_pos_embed:
#             self.pos_embed = HyperbolicPosEmbed(num_patches + self.num_tokens, embed_dim)
#             self.pos_drop = nn.Dropout(p=drop_rate)
#
#         if if_rope:
#             half_head_dim = embed_dim // 2
#             hw_seq_len = img_size // patch_size
#             self.rope = VisionRotaryEmbeddingFast(
#                 dim=half_head_dim,
#                 pt_seq_len=pt_hw_seq_len,
#                 ft_seq_len=hw_seq_len
#             )
#
#         # 使用双曲测地线距离分类头
#         if num_classes > 0:
#             self.head = HyperbolicDistanceHead(embed_dim, num_classes)
#         else:
#             self.head = nn.Identity()
#
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
#         inter_dpr = [0.0] + dpr
#         self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
#
#         self.layers = nn.ModuleList(
#             [
#                 create_block(
#                     embed_dim,
#                     d_state=d_state,
#                     ssm_cfg=ssm_cfg,
#                     norm_epsilon=norm_epsilon,
#                     rms_norm=rms_norm,
#                     residual_in_fp32=residual_in_fp32,
#                     fused_add_norm=fused_add_norm,
#                     layer_idx=i,
#                     if_bimamba=if_bimamba,
#                     bimamba_type=bimamba_type,
#                     drop_path=inter_dpr[i],
#                     if_divide_out=if_divide_out,
#                     init_layer_scale=init_layer_scale,
#                     **factory_kwargs,
#                 )
#                 for i in range(depth)
#             ]
#         )
#
#         # 双曲归一化
#         self.norm_f = HyperbolicLayerNorm(embed_dim, eps=norm_epsilon)
#
#         self.patch_embed.apply(segm_init_weights)
#
#         if not isinstance(self.head, HyperbolicDistanceHead):
#             self.head.apply(segm_init_weights)
#
#         if if_cls_token:
#             if use_double_cls_token:
#                 trunc_normal_(self.cls_token_head, std=.02)
#                 trunc_normal_(self.cls_token_tail, std=.02)
#             else:
#                 trunc_normal_(self.cls_token, std=.02)
#
#         self.apply(
#             partial(
#                 _init_weights,
#                 n_layer=depth,
#                 **(initializer_cfg if initializer_cfg is not None else {}),
#             )
#         )
#
#     def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
#         return {
#             i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
#             for i, layer in enumerate(self.layers)
#         }
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}
#
#     @torch.jit.ignore()
#     def load_pretrained(self, checkpoint_path, prefix=""):
#         _load_weights(self, checkpoint_path, prefix)
#
#     def forward_features(self, x, inference_params=None, if_random_cls_token_position=False,
#                          if_random_token_rank=False):
#         x = self.patch_embed(x)
#         B, M, _ = x.shape
#
#         if self.if_cls_token:
#             if self.use_double_cls_token:
#                 cls_token_head = self.cls_token_head.expand(B, -1, -1)
#                 cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
#                 token_position = [0, M + 1]
#                 # 双曲空间连接
#                 x = torch.cat((
#                     cls_token_head,
#                     x,
#                     cls_token_tail
#                 ), dim=1)
#             else:
#                 if self.use_middle_cls_token:
#                     cls_token = self.cls_token.expand(B, -1, -1)
#                     token_position = M // 2
#                     x = torch.cat((
#                         x[:, :token_position, :],
#                         cls_token,
#                         x[:, token_position:, :]
#                     ), dim=1)
#                 elif if_random_cls_token_position:
#                     cls_token = self.cls_token.expand(B, -1, -1)
#                     token_position = random.randint(0, M)
#                     x = torch.cat((
#                         x[:, :token_position, :],
#                         cls_token,
#                         x[:, token_position:, :]
#                     ), dim=1)
#                 else:
#                     cls_token = self.cls_token.expand(B, -1, -1)
#                     token_position = 0
#                     x = torch.cat((cls_token, x), dim=1)
#             M = x.shape[1]
#
#         if self.if_abs_pos_embed:
#             x = self.pos_embed(x)
#             x = self.pos_drop(x)
#
#         if if_random_token_rank:
#             shuffle_indices = torch.randperm(M)
#             x = x[:, shuffle_indices, :]
#             if isinstance(token_position, list):
#                 token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in
#                                   range(len(token_position))]
#             else:
#                 token_position = torch.where(shuffle_indices == token_position)[0].item()
#
#         if_flip_img_sequences = False
#         if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
#             x = x.flip([1])
#             if_flip_img_sequences = True
#
#         residual = None
#         hidden_states = x
#         if not self.if_bidirectional:
#             for layer in self.layers:
#                 if if_flip_img_sequences and self.if_rope:
#                     hidden_states = hidden_states.flip([1])
#                     if residual is not None:
#                         residual = residual.flip([1])
#
#                 if self.if_rope:
#                     hidden_states = self.rope(hidden_states)
#                     if residual is not None and self.if_rope_residual:
#                         residual = self.rope(residual)
#
#                 if if_flip_img_sequences and self.if_rope:
#                     hidden_states = hidden_states.flip([1])
#                     if residual is not None:
#                         residual = residual.flip([1])
#
#                 hidden_states, residual = layer(
#                     hidden_states, residual, inference_params=inference_params
#                 )
#         else:
#             for i in range(len(self.layers) // 2):
#                 if self.if_rope:
#                     hidden_states = self.rope(hidden_states)
#                     if residual is not None and self.if_rope_residual:
#                         residual = self.rope(residual)
#
#                 hidden_states_f, residual_f = self.layers[i * 2](
#                     hidden_states, residual, inference_params=inference_params
#                 )
#                 hidden_states_b, residual_b = self.layers[i * 2 + 1](
#                     hidden_states.flip([1]), None if residual == None else residual.flip([1]),
#                     inference_params=inference_params
#                 )
#                 # 双曲空间加法
#                 hidden_states = self.manifold.mobius_add(hidden_states_f, hidden_states_b.flip([1]))
#                 residual = self.manifold.mobius_add(residual_f, residual_b.flip([1]))
#
#         if not self.fused_add_norm:
#             if residual is None:
#                 residual = hidden_states
#             else:
#                 # 双曲空间残差连接
#                 residual = self.manifold.mobius_add(residual, self.drop_path(hidden_states))
#             hidden_states = self.norm_f(residual)
#         else:
#             raise NotImplementedError
#
#         if self.if_cls_token:
#             if self.use_double_cls_token:
#                 return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
#             else:
#                 return hidden_states[:, token_position, :]
#
#         if self.final_pool_type == 'none':
#             return hidden_states[:, -1, :]
#         elif self.final_pool_type == 'mean':
#             # 双曲平均
#             return self.manifold.karcher_mean(hidden_states, dim=1)
#         elif self.final_pool_type == 'max':
#             return hidden_states
#         elif self.final_pool_type == 'all':
#             return hidden_states
#         else:
#             raise NotImplementedError
#
#     def forward(self, x, return_features=False, inference_params=None, if_random_cls_token_position=False,
#                 if_random_token_rank=False):
#         x = self.forward_features(x, inference_params, if_random_cls_token_position=if_random_cls_token_position,
#                                   if_random_token_rank=if_random_token_rank)
#         if return_features:
#             return x
#         x = self.head(x)
#         if self.final_pool_type == 'max':
#             x = x.max(dim=1)[0]
#         return x
#
# # 注册双曲Mamba模型
# @register_model
# def hyper_vim_tiny_patch16_224(pretrained=False, **kwargs):
#     model = VisionMamba(
#         patch_size=16, embed_dim=256, depth=12, rms_norm=False, residual_in_fp32=False, fused_add_norm=False,
#         final_pool_type='mean', if_abs_pos_embed=True, if_rope=False,
#         bimamba_type="v2", if_cls_token=True, use_middle_cls_token=True, **kwargs)
#     model.default_cfg = _cfg()
#     return model
#
# @register_model
# def hyper_vim_small_patch16_224(pretrained=False, **kwargs):
#     model = VisionMamba(
#         patch_size=16, embed_dim=384, depth=24, rms_norm=False,
#         final_pool_type='mean', if_abs_pos_embed=True, if_rope=False,
#         bimamba_type="v2", if_cls_token=True, use_middle_cls_token=True, **kwargs)
#     model.default_cfg = _cfg()
#     return model
#
# @register_model
# def hyper_vim_base_patch16_224(pretrained=False, **kwargs):
#     model = VisionMamba(
#         patch_size=16, embed_dim=768, depth=24, rms_norm=False,
#         final_pool_type='mean', if_abs_pos_embed=True, if_rope=False,
#         bimamba_type="v2", if_cls_token=True, use_middle_cls_token=True, **kwargs)
#     model.default_cfg = _cfg()
#     return model

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from functools import partial
# from torch import Tensor
# from typing import Optional
#
# import geoopt
# from geoopt import ManifoldParameter
# from timm.models.vision_transformer import VisionTransformer, _cfg
# from timm.models.registry import register_model
# from timm.models.layers import trunc_normal_, lecun_normal_
#
# from timm.models.layers import DropPath, to_2tuple
# from timm.models.vision_transformer import _load_weights
#
# import math
#
# from collections import namedtuple
#
# from mamba_ssm.modules.mamba_simple import Mamba
# from mamba_ssm.utils.generation import GenerationMixin
# from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
#
# from rope import *
# import random
#
# try:
#     from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
# except ImportError:
#     RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
#
# __all__ = [
#     'vim_tiny_patch16_224', 'vim_small_patch16_224', 'vim_base_patch16_224',
#     'vim_tiny_patch16_384', 'vim_small_patch16_384', 'vim_base_patch16_384',
# ]
#
# GLOBAL_MANIFOLD = geoopt.PoincareBall(c=0.5)
#
# class StablePoincareBall(geoopt.PoincareBall):
#     def __init__(self, c=0.5):
#         super().__init__(c=c)
#     def expmap0(self, x, u):
#         # 添加梯度裁剪防止NaN
#         u_norm = u.norm(dim=-1, keepdim=True).clamp(min=1e-8, max=5.0)
#         return super().expmap0(x, u / u_norm * u_norm.clamp(max=self.radius*0.9))
#
#     # def logmap(self, x, y):
#     #     # 添加安全除法
#     #     diff = super().logmap(x, y)
#         # norm = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)
#         # return diff / (diff.norm(dim=-1, keepdim=True).clamp(min=1e-8))
#         # return diff
#         # return diff / norm * norm.clamp(max=5.0)
#     def logmap0(self, y: torch.Tensor) -> torch.Tensor:
#         # 添加安全截断
#         y_norm = y.norm(dim=-1, keepdim=True).clamp(max=0.95 * (1 / torch.sqrt(self.c)))
#         return super().logmap0(y / y_norm * y_norm)  # 确保在安全范围内
#
# class HyperbolicLogisticRegression(nn.Module):
#     def __init__(self, in_features, num_classes):
#         super().__init__()
#         self.manifold = GLOBAL_MANIFOLD
#         self.num_classes = num_classes
#         # 增加中间层增强梯度传播
#         self.proj = nn.Linear(in_features, in_features)  # 欧氏空间投影
#         self.weight = nn.Parameter(torch.zeros(num_classes, in_features))
#         self.bias = nn.Parameter(torch.zeros(num_classes))
#         nn.init.kaiming_normal_(self.proj.weight, nonlinearity='relu')
#         nn.init.normal_(self.weight, std=0.01)
#         nn.init.zeros_(self.bias)
#         self.grad_scale = nn.Parameter(torch.tensor(1.0))  # 梯度缩放因子
#
#     def forward(self, x):
#         x_tangent = self.manifold.logmap0(x)
#         x_proj = F.relu(self.proj(x_tangent))  # 增加非线性变换
#         # 缩放梯度流
#         x_proj = x_proj * self.grad_scale + x_tangent  # 残差连接
#         logits = F.linear(x_proj, self.weight, self.bias)
#         return logits
# # class HyperbolicLogisticRegression(nn.Module):
# #     """双曲逻辑回归分类头"""
# #
# #     def __init__(self, in_features, num_classes, manifold):
# #         super().__init__()
# #         self.manifold = manifold
# #         self.num_classes = num_classes
# #
# #         # 可学习的映射矩阵和偏置
# #         self.weight = nn.Parameter(torch.zeros(num_classes, in_features))
# #         self.bias = nn.Parameter(torch.zeros(num_classes))
# #         nn.init.normal_(self.weight, std=0.01)
# #         nn.init.zeros_(self.bias)
# #
# #     def forward(self, x):
# #         # x: [B, D] 双曲空间特征
# #         # 映射到切空间
# #         x_tangent = self.manifold.logmap0(x)  # [B, D]
# #
# #         # 计算logits (标准线性变换)
# #         logits = F.linear(x_tangent, self.weight, self.bias)
# #         return logits
#
# # # models_mamba.py 中添加新分类头
# # class HyperbolicLogisticRegression(nn.Module):
# #     """双曲逻辑回归分类头"""
# #
# #     def __init__(self, in_features, num_classes, manifold):
# #         super().__init__()
# #         self.manifold = manifold
# #         self.num_classes = num_classes
# #
# #         # 可学习的映射矩阵和偏置
# #         self.weight = ManifoldParameter(
# #             torch.zeros(num_classes, in_features),
# #             manifold=manifold
# #         )
# #         self.bias = ManifoldParameter(
# #             torch.zeros(num_classes),
# #             manifold=manifold
# #         )
# #         nn.init.normal_(self.weight, std=0.01)
# #         nn.init.zeros_(self.bias)
# #
# #     def forward(self, x):
# #         # x: [B, D] 双曲空间特征
# #         # 映射到切空间
# #         x_tangent = self.manifold.logmap0(x)  # [B, D]
# #
# #         # 计算logits
# #         logits = F.linear(x_tangent, self.weight.tan, self.bias.tan)
# #         return logits
#
# class HyperbolicDistanceHead(nn.Module):
#     """双曲测地线距离分类头"""
#
#     def __init__(self, in_features, num_classes):
#         super().__init__()
#         self.manifold = GLOBAL_MANIFOLD
#         self.num_classes = num_classes
#
#         # # 可学习的类别原型（在双曲空间中）
#         # self.prototypes = ManifoldParameter(
#         #     torch.zeros(num_classes, in_features),
#         #     manifold=manifold
#         # )
#         # nn.init.normal_(self.prototypes, std=0.01)  # 小范围初始化
#
#         # 新初始化方式：从切空间映射，确保稀疏分布
#         init_tangent = torch.randn(num_classes, in_features) * 0.1  # 增大标准差
#         self.prototypes = ManifoldParameter(GLOBAL_MANIFOLD.expmap0(init_tangent), manifold=GLOBAL_MANIFOLD)
#         # 强制约束在双曲空间内（范数 < 1/sqrt(c)）
#         with torch.no_grad():
#             max_norm = 1.0 / torch.sqrt(torch.tensor(GLOBAL_MANIFOLD.c)) - 1e-5
#             norm = self.prototypes.norm(dim=-1, keepdim=True)
#             self.prototypes.data = self.prototypes.data * (max_norm / norm.clamp(min=1e-8))
#
#     def forward(self, x):
#         """
#         计算输入与类别原型的负双曲测地线距离
#         Args:
#             x: 输入特征 [B, D] (欧几里得空间)
#         Returns:
#             logits: 负测地线距离 [B, num_classes]
#         """
#         # 将输入映射到双曲空间
#         # x_hyp = self.manifold.expmap0(x)  # [B, D]
#         x_hyp = x
#
#         # 扩展维度用于广播计算
#         x_hyp = x_hyp.unsqueeze(1)  # [B, 1, D]
#         prototypes = self.prototypes.unsqueeze(0)  # [1, C, D]
#
#         # 计算测地线距离 (使用PoincareBall的dist方法)
#         distances = self.manifold.dist(x_hyp, prototypes)  # [B, C]
#         logits = -distances
#
#         # # 返回负距离作为logits
#         # return -distances
#         return logits
#
#
# # 双曲层归一化
# class HyperbolicLayerNorm(nn.Module):
#     def __init__(self, embedding_dim, eps=1e-5):
#         super().__init__()
#         self.manifold = GLOBAL_MANIFOLD
#         self.eps = eps
#         self.gamma = nn.Parameter(torch.ones(embedding_dim))
#         self.beta = nn.Parameter(torch.zeros(embedding_dim))
#
#     def forward(self, x):
#         # 映射到原点切空间
#         x_tangent = self.manifold.logmap0(x)
#         # 应用层归一化
#         mean = x_tangent.mean(dim=-1, keepdim=True)
#         std = x_tangent.std(dim=-1, keepdim=True)
#         x_norm = (x_tangent - mean) / (std + self.eps)
#         x_norm = x_norm * self.gamma + self.beta
#         # 映射回双曲空间
#         return self.manifold.expmap0(x_norm)
#
#
# # 双曲线性层
# class HyperbolicLinear(nn.Module):
#     def __init__(self, in_features, out_features, bias=True):
#         super().__init__()
#         self.manifold = GLOBAL_MANIFOLD
#         self.in_features = in_features
#         self.out_features = out_features
#
#         self.weight = ManifoldParameter(
#             torch.Tensor(out_features, in_features), manifold=GLOBAL_MANIFOLD
#         )
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if bias:
#             self.bias = ManifoldParameter(
#                 torch.zeros(out_features), manifold=GLOBAL_MANIFOLD
#             )
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.bias, -bound, bound)
#         else:
#             self.bias = None
#
#     def forward(self, input):
#         # 双曲矩阵乘法
#         output = self.manifold.mobius_matvec(self.weight, input)
#         if self.bias is not None:
#             # 双曲加法
#             output = self.manifold.mobius_add(output, self.bias)
#         return output
#
#
# # 双曲位置编码
# class HyperbolicPosEmbed(nn.Module):
#     def __init__(self, num_patches, embed_dim, manifold=None):
#         super().__init__()
#         self.manifold = manifold if manifold is not None else GLOBAL_MANIFOLD
#         self.pos_embed = ManifoldParameter(
#             torch.zeros(1, num_patches, embed_dim), manifold=GLOBAL_MANIFOLD
#         )
#         trunc_normal_(self.pos_embed, std=.02)
#
#     def forward(self, x):
#         # 双曲加法
#         return self.manifold.mobius_add(x, self.pos_embed)
#
#
# # 双曲Patch Embedding
# class HyperbolicPatchEmbed(nn.Module):
#     def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, flatten=True):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#         self.flatten = flatten
#         self.manifold = GLOBAL_MANIFOLD
#
#         # 使用双曲线性层替代卷积
#         # self.proj = nn.Sequential(
#         #     nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride),
#         #     nn.Flatten(2) if flatten else nn.Identity(),
#         #     nn.Linear((patch_size[0] * patch_size[1] * in_chans), embed_dim)  # 先投影到欧几里得空间
#         # )
#
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
#         self.flatten = nn.Flatten(2) if flatten else nn.Identity()
#
#         self.hyperbolic_proj = HyperbolicLinear(embed_dim, embed_dim, self.manifold)
#         self.norm = HyperbolicLayerNorm(embed_dim, self.manifold)
#
#     # def forward(self, x):
#     #     B, C, H, W = x.shape
#     #     assert H == self.img_size[0] and W == self.img_size[1], \
#     #         f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#     #     x = self.proj(x)  # 欧几里得空间投影
#     #     # if self.flatten:
#     #     #     x = x.transpose(1, 2)  # BCHW -> BNC
#     #
#     #     x = self.flatten(x)
#     #     if isinstance(self.flatten, nn.Flatten):
#     #         x = x.transpose(1, 2)
#     #
#     #     x = self.manifold.expmap0(x)  # 映射到双曲空间
#     #     x = self.hyperbolic_proj(x)  # 双曲空间投影
#     #     x = self.norm(x)
#     #     return x
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.proj(x)  # 欧几里得空间投影
#         x = self.flatten(x)
#         if isinstance(self.flatten, nn.Flatten):
#             x = x.transpose(1, 2)
#
#         # 添加特征裁剪（在指数映射前）
#         if self.manifold is not None:  # 仅双曲模式需要裁剪
#             r = 0.8  # 裁剪半径，根据论文建议设为1.0
#             norm = torch.norm(x, p=2, dim=-1, keepdim=True)
#             # 防止除零
#             norm = torch.clamp(norm, min=1e-8)
#             # 裁剪条件：范数大于r时缩放至r
#             scale = torch.where(norm > r, r / norm, torch.ones_like(norm))
#             x = x * scale
#
#         x = self.manifold.expmap0(x)  # 映射到双曲空间
#         x = self.hyperbolic_proj(x)  # 双曲空间投影
#         x = self.norm(x)
#         return x
#
#
# class Block(nn.Module):
#     def __init__(
#             self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.):
#         super().__init__()
#         self.residual_in_fp32 = residual_in_fp32
#         self.fused_add_norm = fused_add_norm
#         self.manifold = GLOBAL_MANIFOLD
#         self.mixer = mixer_cls(dim)
#         self.norm = norm_cls(dim)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         if self.fused_add_norm:
#             assert RMSNorm is not None, "RMSNorm import fails"
#             assert isinstance(
#                 self.norm, (nn.LayerNorm, RMSNorm)
#             ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
#
#     def forward(
#             self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
#     ):
#
#
#
#         if not self.fused_add_norm:
#             if residual is None:
#                 residual = hidden_states
#             else:
#                 # 双曲空间残差连接
#                 residual = self.manifold.mobius_add(residual, self.drop_path(hidden_states))
#
#             # 双曲层归一化
#             hidden_states = self.norm(residual)
#             if self.residual_in_fp32:
#                 residual = residual.to(torch.float32)
#         else:
#             # 不支持融合操作
#             raise NotImplementedError("Fused add-norm not supported in hyperbolic space")
#
#         # Mamba操作在切空间进行
#         hidden_states_tangent = self.manifold.logmap0(hidden_states)
#         hidden_states_tangent = self.mixer(hidden_states_tangent, inference_params=inference_params)
#         hidden_states = self.manifold.expmap0(hidden_states_tangent)
#
#         return hidden_states, residual
#
#     def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
#         return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
#
#
# def create_block(
#         d_model,
#         d_state=16,
#         ssm_cfg=None,
#         norm_epsilon=1e-5,
#         drop_path=0.,
#         rms_norm=False,
#         residual_in_fp32=False,
#         fused_add_norm=False,
#         layer_idx=None,
#         device=None,
#         dtype=None,
#         if_bimamba=False,
#         bimamba_type="none",
#         if_divide_out=False,
#         init_layer_scale=None,
# ):
#     if if_bimamba:
#         bimamba_type = "v1"
#     if ssm_cfg is None:
#         ssm_cfg = {}
#     factory_kwargs = {"device": device, "dtype": dtype}
#     mixer_cls = partial(Mamba, d_state=d_state, layer_idx=layer_idx, bimamba_type=bimamba_type,
#                         if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
#
#     # 使用双曲层归一化
#     norm_cls = partial(
#         HyperbolicLayerNorm, eps=norm_epsilon
#     ) if GLOBAL_MANIFOLD else (
#         partial(nn.LayerNorm, eps=norm_epsilon) if not rms_norm else partial(RMSNorm, eps=norm_epsilon)
#     )
#
#     block = Block(
#         d_model,
#         mixer_cls,
#         norm_cls=norm_cls,
#         drop_path=drop_path,
#         fused_add_norm=fused_add_norm,
#         residual_in_fp32=residual_in_fp32,
#     )
#     block.layer_idx = layer_idx
#     return block
#
#
# def _init_weights(
#         module,
#         n_layer,
#         initializer_range=0.02,
#         rescale_prenorm_residual=True,
#         n_residuals_per_layer=1,
# ):
#     if isinstance(module, nn.Linear):
#         if module.bias is not None:
#             if not getattr(module.bias, "_no_reinit", False):
#                 nn.init.zeros_(module.bias)
#     elif isinstance(module, nn.Embedding):
#         nn.init.normal_(module.weight, std=initializer_range)
#
#     if rescale_prenorm_residual:
#         for name, p in module.named_parameters():
#             if name in ["out_proj.weight", "fc2.weight"]:
#                 nn.init.kaiming_uniform_(p, a=math.sqrt(5))
#                 with torch.no_grad():
#                     p /= math.sqrt(n_residuals_per_layer * n_layer)
#
#
# # def segm_init_weights(m):
# #     if isinstance(m, nn.Linear):
# #         trunc_normal_(m.weight, std=0.02)
# #         if isinstance(m, nn.Linear) and m.bias is not None:
# #             nn.init.constant_(m.bias, 0)
# #     elif isinstance(m, nn.Conv2d):
# #         lecun_normal_(m.weight)
# #         if m.bias is not None:
# #             nn.init.zeros_(m.bias)
# #     elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
# #         nn.init.zeros_(m.bias)
# #         nn.init.ones_(m.weight)
# def segm_init_weights(m):
#     if isinstance(m, nn.Linear):
#         # 增大初始化标准差
#         trunc_normal_(m.weight, std=0.02)
#         if isinstance(m, nn.Linear) and m.bias is not None:
#             nn.init.normal_(m.bias, std=0.005)  # 偏置也使用正态初始化
#     elif isinstance(m, nn.Conv2d):
#         lecun_normal_(m.weight)  # 增强卷积层初始化
#         if m.bias is not None:
#             nn.init.normal_(m.bias, std=0.005)
#     elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
#         nn.init.zeros_(m.bias)
#         nn.init.ones_(m.weight)
#     # 双曲参数特殊初始化
#     # elif isinstance(m, ManifoldParameter) and hasattr(m, 'manifold'):
#     #     with torch.no_grad():
#     #         # 确保初始点在空间中心附近
#     #         m.data = m.manifold.expmap0(torch.randn_like(m.data) * 0.05)
#     elif isinstance(m, ManifoldParameter) and hasattr(m, 'manifold'):
#         with torch.no_grad():
#             # 生成初始化点（切空间）
#             init_tangent = torch.randn_like(m.data) * 0.01  # 更小的噪声
#             # 映射到双曲空间
#             m.data = m.manifold.expmap0(init_tangent)
#             # 范数裁剪：确保点不靠近边界
#             max_norm = 0.9 / torch.sqrt(torch.tensor(m.manifold.c))  # 90% 边界
#             norm = m.data.norm(dim=-1, keepdim=True)
#             # 缩放范数过大的点
#             scale = torch.clamp(max_norm / norm, max=1.0)
#             m.data = m.data * scale
#
#
# class VisionMamba(nn.Module):
#     def __init__(self,
#                  img_size=224,
#                  patch_size=16,
#                  stride=16,
#                  depth=24,
#                  embed_dim=192,
#                  d_state=16,
#                  channels=3,
#                  num_classes=1000,
#                  ssm_cfg=None,
#                  drop_rate=0.,
#                  drop_path_rate=0.1,
#                  norm_epsilon: float = 1e-5,
#                  rms_norm: bool = True,
#                  initializer_cfg=None,
#                  fused_add_norm=True,
#                  residual_in_fp32=True,
#                  device=None,
#                  dtype=None,
#                  ft_seq_len=None,
#                  pt_hw_seq_len=14,
#                  if_bidirectional=False,
#                  final_pool_type='none',
#                  if_abs_pos_embed=True,
#                  if_rope=False,
#                  if_rope_residual=False,
#                  flip_img_sequences_ratio=-1.,
#                  if_bimamba=False,
#                  bimamba_type="v2",
#                  if_cls_token=True,
#                  if_divide_out=True,
#                  init_layer_scale=None,
#                  use_double_cls_token=False,
#                  use_middle_cls_token=True,
#                  **kwargs):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         kwargs.update(factory_kwargs)
#         super().__init__()
#         self.residual_in_fp32 = residual_in_fp32
#         self.fused_add_norm = fused_add_norm
#         self.if_bidirectional = if_bidirectional
#         self.final_pool_type = final_pool_type
#         self.if_abs_pos_embed = if_abs_pos_embed
#         self.if_rope = if_rope
#         self.if_rope_residual = if_rope_residual
#         self.flip_img_sequences_ratio = flip_img_sequences_ratio
#         self.if_cls_token = if_cls_token
#         self.use_double_cls_token = use_double_cls_token
#         self.use_middle_cls_token = use_middle_cls_token
#         self.num_tokens = 1 if if_cls_token else 0
#         self.manifold = GLOBAL_MANIFOLD
#
#         self.num_classes = num_classes
#         self.d_model = self.num_features = self.embed_dim = embed_dim
#
#         # 使用双曲Patch Embedding
#         self.patch_embed = HyperbolicPatchEmbed(
#             img_size=img_size, patch_size=patch_size, stride=stride,
#             in_chans=channels, embed_dim=embed_dim
#         )
#         num_patches = self.patch_embed.num_patches
#
#         if if_cls_token:
#             if use_double_cls_token:
#                 self.cls_token_head = ManifoldParameter(torch.zeros(1, 1, embed_dim),
#                                                         manifold=self.manifold)
#                 self.cls_token_tail = ManifoldParameter(torch.zeros(1, 1, embed_dim),
#                                                         manifold=self.manifold)
#                 self.num_tokens = 2
#             else:
#                 self.cls_token = ManifoldParameter(torch.zeros(1, 1, embed_dim),
#                                                    manifold=self.manifold)
#
#         if if_abs_pos_embed:
#
#             self.pos_embed = HyperbolicPosEmbed(num_patches + self.num_tokens, embed_dim, self.manifold)
#
#             self.pos_drop = nn.Dropout(p=drop_rate)
#
#         if if_rope:
#             half_head_dim = embed_dim // 2
#             hw_seq_len = img_size // patch_size
#             self.rope = VisionRotaryEmbeddingFast(
#                 dim=half_head_dim,
#                 pt_seq_len=pt_hw_seq_len,
#                 ft_seq_len=hw_seq_len
#             )
#
#         # # 使用双曲线性层作为分类头
#         # self.head = HyperbolicLinear(embed_dim, num_classes, self.manifold) if hyperbolic and num_classes > 0 else (
#         #     nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
#         # )
#
#         # 使用双曲测地线距离分类头
#         if  num_classes > 0:
#             self.head = HyperbolicDistanceHead(embed_dim, num_classes)
#             # self.head = HyperbolicLogisticRegression(embed_dim, num_classes, self.manifold)
#
#         elif num_classes > 0:
#             self.head = nn.Linear(embed_dim, num_classes)
#         else:
#             self.head = nn.Identity()
#
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
#         inter_dpr = [0.0] + dpr
#         self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
#
#         self.layers = nn.ModuleList(
#             [
#                 create_block(
#                     embed_dim,
#                     d_state=d_state,
#                     ssm_cfg=ssm_cfg,
#                     norm_epsilon=norm_epsilon,
#                     rms_norm=rms_norm,
#                     residual_in_fp32=residual_in_fp32,
#                     fused_add_norm=fused_add_norm,
#                     layer_idx=i,
#                     if_bimamba=if_bimamba,
#                     bimamba_type=bimamba_type,
#                     drop_path=inter_dpr[i],
#                     if_divide_out=if_divide_out,
#                     init_layer_scale=init_layer_scale,
#                     **factory_kwargs,
#                 )
#                 for i in range(depth)
#             ]
#         )
#
#         # 双曲归一化
#         self.norm_f = HyperbolicLayerNorm(embed_dim, eps=norm_epsilon)
#
#         self.patch_embed.apply(segm_init_weights)
#         # self.head.apply(segm_init_weights)
#
#         if not isinstance(self.head, HyperbolicDistanceHead):
#         # if not isinstance(self.head, HyperbolicLogisticRegression):
#             self.head.apply(segm_init_weights)
#
#
#         if if_cls_token:
#             if use_double_cls_token:
#                 trunc_normal_(self.cls_token_head, std=.02)
#                 trunc_normal_(self.cls_token_tail, std=.02)
#             else:
#                 trunc_normal_(self.cls_token, std=.02)
#
#         self.apply(
#             partial(
#                 _init_weights,
#                 n_layer=depth,
#                 **(initializer_cfg if initializer_cfg is not None else {}),
#             )
#         )
#
#     def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
#         return {
#             i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
#             for i, layer in enumerate(self.layers)
#         }
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}
#
#     @torch.jit.ignore()
#     def load_pretrained(self, checkpoint_path, prefix=""):
#         _load_weights(self, checkpoint_path, prefix)
#
#     def forward_features(self, x, inference_params=None, if_random_cls_token_position=False,
#                          if_random_token_rank=False):
#         x = self.patch_embed(x)
#         B, M, _ = x.shape
#
#         if self.if_cls_token:
#             if self.use_double_cls_token:
#                 cls_token_head = self.cls_token_head.expand(B, -1, -1)
#                 cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
#                 token_position = [0, M + 1]
#                 # 双曲空间连接
#                 if self.hyperbolic:
#                     x = torch.cat((
#                         self.manifold.expmap0(cls_token_head),
#                         x,
#                         self.manifold.expmap0(cls_token_tail)
#                     ), dim=1)
#                 else:
#                     x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
#                 M = x.shape[1]
#             else:
#                 if self.use_middle_cls_token:
#                     cls_token = self.cls_token.expand(B, -1, -1)
#                     token_position = M // 2
#                     if self.hyperbolic:
#                         cls_token = self.manifold.expmap0(cls_token)
#                         x = torch.cat((
#                             x[:, :token_position, :],
#                             cls_token,
#                             x[:, token_position:, :]
#                         ), dim=1)
#                     else:
#                         x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
#                 elif if_random_cls_token_position:
#                     cls_token = self.cls_token.expand(B, -1, -1)
#                     token_position = random.randint(0, M)
#                     if self.hyperbolic:
#                         cls_token = self.manifold.expmap0(cls_token)
#                         x = torch.cat((
#                             x[:, :token_position, :],
#                             cls_token,
#                             x[:, token_position:, :]
#                         ), dim=1)
#                     else:
#                         x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
#                 else:
#                     cls_token = self.cls_token.expand(B, -1, -1)
#                     token_position = 0
#                     if self.hyperbolic:
#                         cls_token = self.manifold.expmap0(cls_token)
#                         x = torch.cat((cls_token, x), dim=1)
#                     else:
#                         x = torch.cat((cls_token, x), dim=1)
#                 M = x.shape[1]
#
#         if self.if_abs_pos_embed:
#             if self.hyperbolic:
#                 x = self.pos_embed(x)
#             else:
#                 x = x + self.pos_embed
#             x = self.pos_drop(x)
#
#         if if_random_token_rank:
#             shuffle_indices = torch.randperm(M)
#             x = x[:, shuffle_indices, :]
#             if isinstance(token_position, list):
#                 token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in
#                                   range(len(token_position))]
#             else:
#                 token_position = torch.where(shuffle_indices == token_position)[0].item()
#
#         if_flip_img_sequences = False
#         if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
#             x = x.flip([1])
#             if_flip_img_sequences = True
#
#         residual = None
#         hidden_states = x
#         if not self.if_bidirectional:
#             for layer in self.layers:
#                 if if_flip_img_sequences and self.if_rope:
#                     hidden_states = hidden_states.flip([1])
#                     if residual is not None:
#                         residual = residual.flip([1])
#
#                 if self.if_rope:
#                     hidden_states = self.rope(hidden_states)
#                     if residual is not None and self.if_rope_residual:
#                         residual = self.rope(residual)
#
#                 if if_flip_img_sequences and self.if_rope:
#                     hidden_states = hidden_states.flip([1])
#                     if residual is not None:
#                         residual = residual.flip([1])
#
#                 hidden_states, residual = layer(
#                     hidden_states, residual, inference_params=inference_params
#                 )
#         else:
#             for i in range(len(self.layers) // 2):
#                 if self.if_rope:
#                     hidden_states = self.rope(hidden_states)
#                     if residual is not None and self.if_rope_residual:
#                         residual = self.rope(residual)
#
#                 hidden_states_f, residual_f = self.layers[i * 2](
#                     hidden_states, residual, inference_params=inference_params
#                 )
#                 hidden_states_b, residual_b = self.layers[i * 2 + 1](
#                     hidden_states.flip([1]), None if residual == None else residual.flip([1]),
#                     inference_params=inference_params
#                 )
#                 # 双曲空间加法
#                 if self.hyperbolic:
#                     hidden_states = self.manifold.mobius_add(hidden_states_f, hidden_states_b.flip([1]))
#                     residual = self.manifold.mobius_add(residual_f, residual_b.flip([1]))
#                 else:
#                     hidden_states = hidden_states_f + hidden_states_b.flip([1])
#                     residual = residual_f + residual_b.flip([1])
#
#         if not self.fused_add_norm:
#             if residual is None:
#                 residual = hidden_states
#             else:
#                 # 双曲空间残差连接
#                 if self.hyperbolic:
#                     residual = self.manifold.mobius_add(residual, self.drop_path(hidden_states))
#                 else:
#                     residual = residual + self.drop_path(hidden_states)
#             hidden_states = self.norm_f(residual)
#         else:
#             raise NotImplementedError
#
#         # if self.if_cls_token:
#         #     if self.use_double_cls_token:
#         #         if self.hyperbolic:
#         #             return self.manifold.logmap0((
#         #                                                  hidden_states[:, token_position[0], :] +
#         #                                                  hidden_states[:, token_position[1], :]
#         #                                          ) / 2)
#         #         else:
#         #             return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
#         #     else:
#         #         if self.hyperbolic:
#         #             return self.manifold.logmap0(hidden_states[:, token_position, :])
#         #         else:
#         #             return hidden_states[:, token_position, :]
#
#         if self.if_cls_token:
#             if self.use_double_cls_token:
#                 if self.hyperbolic:
#                     return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
#                 else:
#                     return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
#             else:
#                 if self.hyperbolic:
#                     return hidden_states[:, token_position, :]
#                 else:
#                     return hidden_states[:, token_position, :]
#
#         if self.final_pool_type == 'none':
#             if self.hyperbolic:
#                 return self.manifold.logmap0(hidden_states[:, -1, :])
#             else:
#                 return hidden_states[:, -1, :]
#         elif self.final_pool_type == 'mean':
#             if self.hyperbolic:
#                 # 双曲平均
#                 return self.manifold.logmap0(self.manifold.karcher_mean(hidden_states, dim=1))
#             else:
#                 return hidden_states.mean(dim=1)
#         elif self.final_pool_type == 'max':
#             return hidden_states
#         elif self.final_pool_type == 'all':
#             return hidden_states
#         else:
#             raise NotImplementedError
#
#     def forward(self, x, return_features=False, inference_params=None, if_random_cls_token_position=False,
#                 if_random_token_rank=False):
#         x = self.forward_features(x, inference_params, if_random_cls_token_position=if_random_cls_token_position,
#                                   if_random_token_rank=if_random_token_rank)
#         if return_features:
#             return x
#         x = self.head(x)
#         if self.final_pool_type == 'max':
#             x = x.max(dim=1)[0]
#         return x
#
#
# # 注册双曲Mamba模型
# @register_model
# def hyper_vim_tiny_patch16_224(pretrained=False, **kwargs):
#     kwargs.setdefault('hyperbolic', True)
#     kwargs.setdefault('curvature', 0.5)
#     model = VisionMamba(
#         patch_size=16, embed_dim=192, depth=24, rms_norm=False, residual_in_fp32=True, fused_add_norm=False,
#         final_pool_type='mean', if_abs_pos_embed=True, if_rope=False,
#         bimamba_type="v2", if_cls_token=True, use_middle_cls_token=True, **kwargs)
#     model.default_cfg = _cfg()
#     return model
#
#
# @register_model
# def hyper_vim_small_patch16_224(pretrained=False, **kwargs):
#     kwargs.setdefault('hyperbolic', True)
#     kwargs.setdefault('curvature', 0.5)
#     model = VisionMamba(
#         patch_size=16, embed_dim=384, depth=24, rms_norm=False,
#         final_pool_type='mean', if_abs_pos_embed=True, if_rope=False,
#         bimamba_type="v2", if_cls_token=True, use_middle_cls_token=True, **kwargs)
#     model.default_cfg = _cfg()
#     return model
#
#
# @register_model
# def hyper_vim_base_patch16_224(pretrained=False, **kwargs):
#     kwargs.setdefault('hyperbolic', True)
#     kwargs.setdefault('curvature', 0.5)
#     model = VisionMamba(
#         patch_size=16, embed_dim=768, depth=24, rms_norm=False,
#         final_pool_type='mean', if_abs_pos_embed=True, if_rope=False,
#         bimamba_type="v2", if_cls_token=True, use_middle_cls_token=True, **kwargs)
#     model.default_cfg = _cfg()
#     return model