# AreaFlashAttention

## 功能描述

AreaFlashAttention 是一个融合算子，将 YOLOv12 中的 Area Attention 机制与 FlashAttention 技术有机结合，实现高效的自注意力计算。

### 核心特性

- **区域划分**：将特征图沿垂直或水平方向切分为 `l` 个独立区域，每个区域内独立计算注意力
- **复杂度降低**：将自注意力复杂度从 `O(n²)` 降低到 `O(n²/l)`
- **内存优化**：通过 FlashAttention 的分块计算和在线 softmax 算法，避免物化完整的注意力矩阵
- **多核并行**：区域之间无数据依赖，天然支持多核并行加速

## 接口定义

```python
torch.ops.ops_transformer.area_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    l: int = 4,
    direction: int = 0,
    scale: float = 1.0,
    layout: int = 0
) -> torch.Tensor
```

## 参数说明

### 输入参数

| 参数名 | 类型 | 必选/可选 | 说明 |
|--------|------|----------|------|
| `query` | torch.Tensor | 必选 | Query 张量，形状为 `[B, N, S, D]` 或 `[B, H, W, C]` |
| `key` | torch.Tensor | 必选 | Key 张量，形状与 `query` 相同 |
| `value` | torch.Tensor | 必选 | Value 张量，形状与 `query` 相同 |
| `l` | int | 可选 | 区域切分数量，默认为 4。表示将序列切分为 `l` 个独立区域 |
| `direction` | int | 可选 | 切分方向，默认为 0。0 表示垂直切分（沿序列维度），1 表示水平切分 |
| `scale` | float | 可选 | 注意力分数的缩放因子，默认为 1.0。通常设置为 `1/sqrt(D)` |
| `layout` | int | 可选 | 输入布局，默认为 0。0 表示 BSND 布局，1 表示 BNSD 布局 |

### 返回值

| 类型 | 说明 |
|------|------|
| torch.Tensor | 注意力输出张量，形状与 `query` 相同 |

## 支持的数据类型

| 数据类型 | 设备支持 | 说明 |
|---------|---------|------|
| torch.float16 | Atlas A2/A3 | 推荐，性能最优 |
| torch.float32 | Atlas A2/A3 | 高精度场景 |
| torch.bfloat16 | Atlas A2/A3 | 训练场景 |

## Shape 约束

### 输入 Shape

- `query`, `key`, `value` 的 shape 必须相同
- 输入必须是 4D 张量：`[B, N, S, D]` 或 `[B, H, W, C]`
  - `B`：batch size
  - `N`：注意力头数（num_heads）
  - `S`：序列长度（sequence length）
  - `D`：头维度（head dimension），必须为 16 的倍数

### 输出 Shape

- 输出 shape 与输入 `query` 相同：`[B, N, S, D]`

### 参数约束

- `l`：区域数量，必须满足 `1 ≤ l ≤ S`，且 `S` 必须能被 `l` 整除
- `direction`：切分方向，必须为 0 或 1
- `scale`：缩放因子，必须大于 0

## 约束条件

1. **序列长度约束**：序列长度 `S` 必须能被区域数量 `l` 整除
2. **头维度约束**：头维度 `D` 必须为 16 的倍数（对齐 Cube 计算单元）
3. **数据类型约束**：`query`, `key`, `value` 的数据类型必须相同
4. **设备约束**：输入张量必须在 NPU 设备上

## 使用示例

### 基本用法

```python
import torch
import torch_npu

# 设置设备
device = torch.device("npu:0")

# 定义输入参数
batch_size = 2
num_heads = 8
seq_len = 512
head_dim = 64
l = 4  # 区域数量

# 创建输入张量
query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)

# 调用算子
output = torch.ops.ops_transformer.area_flash_attention(
    query, key, value, 
    l=l, 
    direction=0, 
    scale=1.0 / (head_dim ** 0.5)
)

print(f"Output shape: {output.shape}")  # [2, 8, 512, 64]
```

### 视觉场景（YOLOv12）

```python
import torch
import torch_npu

device = torch.device("npu:0")

# 模拟 YOLOv12 的特征图
# 假设特征图大小为 14x14，通道数为 512
batch_size = 1
height = 14
width = 14
channels = 512

# 将特征图转换为注意力格式 [B, N, S, D]
# 这里 N=1（单头注意力），S=H*W，D=C
seq_len = height * width  # 196
num_heads = 1
head_dim = channels

query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)

# 使用区域注意力（l=4，将 14x14 切分为 4 个区域）
output = torch.ops.ops_transformer.area_flash_attention(
    query, key, value,
    l=4,
    direction=0,  # 垂直切分
    scale=1.0 / (head_dim ** 0.5)
)

print(f"Output shape: {output.shape}")  # [1, 1, 196, 512]
```

### 不同区域数量对比

```python
import torch
import torch_npu

device = torch.device("npu:0")
batch_size, num_heads, seq_len, head_dim = 1, 8, 512, 64

query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)

# l=1：退化为标准 FlashAttention
output_1 = torch.ops.ops_transformer.area_flash_attention(query, key, value, l=1)

# l=4：标准区域注意力（推荐）
output_4 = torch.ops.ops_transformer.area_flash_attention(query, key, value, l=4)

# l=8：更细粒度的区域划分
output_8 = torch.ops.ops_transformer.area_flash_attention(query, key, value, l=8)

print(f"l=1 output shape: {output_1.shape}")
print(f"l=4 output shape: {output_4.shape}")
print(f"l=8 output shape: {output_8.shape}")
```

## 性能特点

### 计算复杂度

| 配置 | 复杂度 | 说明 |
|------|--------|------|
| 标准 Attention | `O(S²)` | 全局注意力 |
| Area Attention (l=4) | `O(S²/4)` | 复杂度降低 75% |
| Area Attention (l=8) | `O(S²/8)` | 复杂度降低 87.5% |

### 内存访问优化

- **HBM 访问减少**：通过 FlashAttention 的分块计算，避免物化完整的 `S×S` 注意力矩阵
- **UB 利用率高**：根据 UB 容量自动选择最优分块大小
- **多核并行**：每个区域分配给独立的 AI Core，实现线性加速

## 注意事项

1. **输入必须连续**：输入张量必须是连续内存布局
2. **设备一致性**：所有输入张量必须在同一个 NPU 设备上
3. **数值精度**：float16 的精度阈值为最大绝对误差 < 1e-3，余弦相似度 > 0.999
4. **性能调优**：建议根据实际序列长度和头维度调整区域数量 `l` 以获得最佳性能

## 相关算子

- `flash_attention_score`：标准 FlashAttention 算子
- `area_attention_reshape`：区域注意力重塑算子

## 版本信息

- **算子版本**：1.0.0
- **支持平台**：Atlas A2 (ascend910b), Atlas A3 (ascend910_93)
- **CANN 版本**：≥ 8.0.0
