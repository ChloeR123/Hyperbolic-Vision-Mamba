import torch
import geoopt
import pytest
from models_mamba import HyperbolicLayerNorm, HyperbolicLinear, HyperbolicPosEmbed, HyperbolicPatchEmbed, VisionMamba


# 测试基础：创建双曲流形和随机数据
@pytest.fixture
def hyperbolic_setup():
    manifold = geoopt.PoincareBall(c=1.0)
    batch_size, seq_len, embed_dim = 2, 10, 16

    # 生成合法双曲点（范数<1）
    euclidean_points = torch.rand(batch_size, seq_len, embed_dim) * 0.9
    valid_points = manifold.expmap0(euclidean_points)

    # 边界点（范数接近1）
    boundary_point = torch.zeros(1, 1, embed_dim)
    boundary_point[0, 0, 0] = 0.999
    boundary_point = manifold.expmap0(boundary_point)

    # 创建需要梯度的点
    euclidean_points.requires_grad = True

    return {
        'manifold': manifold,
        'valid_points': valid_points,
        'boundary_point': boundary_point,
        'euclidean_points': euclidean_points,  # 用于梯度测试
        'invalid_point': torch.rand(batch_size, seq_len, embed_dim) * 1.1  # 范数>1
    }


# 测试双曲层归一化
def test_hyperbolic_layernorm(hyperbolic_setup):
    manifold = hyperbolic_setup['manifold']
    valid_points = hyperbolic_setup['valid_points']

    # 基础功能测试
    norm_layer = HyperbolicLayerNorm(embedding_dim=16, manifold=manifold)
    output = norm_layer(valid_points)

    # 验证输出仍在流形上
    assert manifold.check_point(output), "Output not on manifold"

    # 验证数值稳定性
    assert not torch.isnan(output).any(), "NaN values detected"
    assert not torch.isinf(output).any(), "Inf values detected"


# 测试双曲线性层
def test_hyperbolic_linear(hyperbolic_setup):
    manifold = hyperbolic_setup['manifold']
    valid_points = hyperbolic_setup['valid_points']

    # 正向传播测试
    linear = HyperbolicLinear(16, 8, manifold)
    output = linear(valid_points)

    # 验证输出维度和流形约束
    assert output.shape == (2, 10, 8), "Incorrect output shape"
    assert manifold.check_point(output), "Output not on manifold"

    # 无偏置测试
    linear_no_bias = HyperbolicLinear(16, 8, manifold, bias=False)
    output_no_bias = linear_no_bias(valid_points)
    assert manifold.check_point(output_no_bias), "Bias-free output not on manifold"

    # 梯度测试
    linear = HyperbolicLinear(16, 8, manifold)
    valid_points.requires_grad = True
    output = linear(valid_points)
    loss = output.norm()
    loss.backward()

    # 验证梯度存在
    assert valid_points.grad is not None, "No gradient for input"
    assert torch.isfinite(valid_points.grad).all(), "Invalid gradients for input"

    # 验证参数梯度
    assert linear.weight.grad is not None, "No gradient for weight"
    assert torch.isfinite(linear.weight.grad).all(), "Invalid gradients for weight"

    if linear.bias is not None:
        assert linear.bias.grad is not None, "No gradient for bias"
        assert torch.isfinite(linear.bias.grad).all(), "Invalid gradients for bias"


# 测试双曲位置编码
def test_hyperbolic_pos_embed(hyperbolic_setup):
    manifold = hyperbolic_setup['manifold']
    valid_points = hyperbolic_setup['valid_points']

    # 初始化位置编码
    pos_embed = HyperbolicPosEmbed(num_patches=10, embed_dim=16, manifold=manifold)
    output = pos_embed(valid_points)

    # 验证输出流形约束
    assert manifold.check_point(output), "Output not on manifold after position embedding"

    # 验证形状一致性
    assert output.shape == valid_points.shape, "Shape mismatch after position embedding"


# 测试双曲Patch Embedding
def test_hyperbolic_patch_embed():
    manifold = geoopt.PoincareBall(c=1.0)
    patch_embed = HyperbolicPatchEmbed(
        img_size=32,
        patch_size=16,
        stride=16,
        in_chans=3,
        embed_dim=16,
        manifold=manifold
    )

    # 创建测试图像
    test_image = torch.rand(2, 3, 32, 32)
    output = patch_embed(test_image)

    # 验证输出形状和流形约束
    assert output.shape == (2, 4, 16), f"Unexpected output shape: {output.shape}"
    assert manifold.check_point(output), "Patch embeddings not on manifold"


# 测试完整模型的双曲操作
def test_vision_mamba_hyperbolic():
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU-dependent test")

    # 创建双曲配置模型 - 显式禁用fused_add_norm
    model = VisionMamba(
        img_size=224,
        patch_size=16,
        embed_dim=16,
        depth=2,
        num_classes=10,
        hyperbolic=True,
        curvature=0.5,
        fused_add_norm=False  # 关键修复：双曲模式下禁用
    ).cuda()  # 将模型移到GPU

    # 测试数据
    test_input = torch.rand(2, 3, 224, 224).cuda()  # 将输入移到GPU

    # 前向传播
    output = model(test_input)

    # 验证输出形状
    assert output.shape == (2, 10), f"Unexpected output shape: {output.shape}"

    # 验证分类头输出在欧式空间（logmap后）
    assert torch.isfinite(output).all(), "Model output contains non-finite values"



# 测试层归一化的梯度计算
def test_hyperbolic_layernorm_grad(hyperbolic_setup):
    manifold = hyperbolic_setup['manifold']
    euclidean_points = hyperbolic_setup['euclidean_points']  # 使用欧几里得点

    # 创建可微分层
    norm_layer = HyperbolicLayerNorm(embedding_dim=16, manifold=manifold)

    # 设置梯度检查
    euclidean_points.requires_grad = True

    # 前向计算：从欧几里得空间映射到双曲空间
    hyperbolic_points = manifold.expmap0(euclidean_points)
    output = norm_layer(hyperbolic_points)

    # 反向传播
    loss = output.norm()
    loss.backward()

    # 验证梯度存在且有限
    assert euclidean_points.grad is not None, "No gradient for input"
    assert torch.isfinite(euclidean_points.grad).all(), "Invalid gradients for input"

    # 检查参数梯度
    assert norm_layer.gamma.grad is not None, "No gradient for gamma"
    assert torch.isfinite(norm_layer.gamma.grad).all(), "Invalid gamma gradient"

    assert norm_layer.beta.grad is not None, "No gradient for beta"
    assert torch.isfinite(norm_layer.beta.grad).all(), "Invalid beta gradient"


# 测试边界情况处理
def test_boundary_handling(hyperbolic_setup):
    manifold = hyperbolic_setup['manifold']
    boundary_point = hyperbolic_setup['boundary_point']

    # 边界点应能通过层处理而不崩溃
    norm_layer = HyperbolicLayerNorm(embedding_dim=16, manifold=manifold)
    linear = HyperbolicLinear(16, 16, manifold)

    # 前向传播（应成功）
    norm_output = norm_layer(boundary_point)
    linear_output = linear(norm_output)

    # 验证输出仍在流形上
    assert manifold.check_point(norm_output), "Norm output exceeded manifold after boundary input"
    assert manifold.check_point(linear_output), "Linear output exceeded manifold after boundary input"


if __name__ == "__main__":
    pytest.main(["-v", "test_hyperbolic_modules.py"])