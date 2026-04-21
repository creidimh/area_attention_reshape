# AreaFlashAttention 测试用例文档

## 1. 支持的数据类型

```python
SUPPORTED_DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
]
```

## 2. 常规测试 Shape（TEST_SHAPES）

这些 shape 覆盖了常见的使用场景：

```python
TEST_SHAPES = [
    # [B, N, S, D] - 小规模
    {"batch_size": 1, "num_heads": 1, "seq_len": 64, "head_dim": 64, "l": 2, "direction": 0},
    {"batch_size": 1, "num_heads": 2, "seq_len": 128, "head_dim": 64, "l": 4, "direction": 0},
    {"batch_size": 2, "num_heads": 4, "seq_len": 256, "head_dim": 64, "l": 4, "direction": 0},
    
    # [B, N, S, D] - 中等规模
    {"batch_size": 1, "num_heads": 8, "seq_len": 512, "head_dim": 64, "l": 4, "direction": 0},
    {"batch_size": 2, "num_heads": 8, "seq_len": 512, "head_dim": 64, "l": 8, "direction": 0},
    {"batch_size": 4, "num_heads": 8, "seq_len": 512, "head_dim": 64, "l": 4, "direction": 1},
    
    # [B, N, S, D] - 大规模
    {"batch_size": 1, "num_heads": 16, "seq_len": 1024, "head_dim": 64, "l": 4, "direction": 0},
    {"batch_size": 2, "num_heads": 16, "seq_len": 1024, "head_dim": 64, "l": 8, "direction": 1},
    
    # [B, N, S, D] - 不同 head_dim
    {"batch_size": 2, "num_heads": 8, "seq_len": 512, "head_dim": 32, "l": 4, "direction": 0},
    {"batch_size": 2, "num_heads": 8, "seq_len": 512, "head_dim": 128, "l": 4, "direction": 0},
    
    # [B, H, W, C] - 视觉场景（YOLOv12）
    {"batch_size": 1, "num_heads": 1, "seq_len": 196, "head_dim": 64, "l": 4, "direction": 0},  # 14x14
    {"batch_size": 1, "num_heads": 1, "seq_len": 784, "head_dim": 64, "l": 4, "direction": 1},  # 28x28
    {"batch_size": 1, "num_heads": 1, "seq_len": 3136, "head_dim": 64, "l": 4, "direction": 0}, # 56x56
]
```

## 3. 泛化测试 Shape（GENERAL_SHAPES）

这些 shape 用于测试算子的泛化能力：

```python
GENERAL_SHAPES = [
    # 不同 batch_size
    {"batch_size": 8, "num_heads": 8, "seq_len": 256, "head_dim": 64, "l": 4, "direction": 0},
    {"batch_size": 16, "num_heads": 4, "seq_len": 128, "head_dim": 64, "l": 2, "direction": 1},
    
    # 不同 num_heads
    {"batch_size": 2, "num_heads": 1, "seq_len": 512, "head_dim": 64, "l": 4, "direction": 0},
    {"batch_size": 2, "num_heads": 32, "seq_len": 256, "head_dim": 64, "l": 4, "direction": 1},
    
    # 不同 seq_len
    {"batch_size": 2, "num_heads": 8, "seq_len": 32, "head_dim": 64, "l": 2, "direction": 0},
    {"batch_size": 2, "num_heads": 8, "seq_len": 2048, "head_dim": 64, "l": 8, "direction": 1},
    
    # 不同 l（区域数量）
    {"batch_size": 2, "num_heads": 8, "seq_len": 512, "head_dim": 64, "l": 1, "direction": 0},
    {"batch_size": 2, "num_heads": 8, "seq_len": 512, "head_dim": 64, "l": 16, "direction": 1},
    
    # 不同 direction
    {"batch_size": 2, "num_heads": 8, "seq_len": 512, "head_dim": 64, "l": 2, "direction": 0},
    {"batch_size": 2, "num_heads": 8, "seq_len": 512, "head_dim": 64, "l": 2, "direction": 1},
]
```

## 4. 边界值测试（BOUNDARY_VALUES）

这些用例测试边界条件和特殊场景：

```python
BOUNDARY_VALUES = [
    # 最小序列长度
    {"batch_size": 1, "num_heads": 1, "seq_len": 4, "head_dim": 16, "l": 2, "direction": 0},
    
    # 最小区域数量（l=1，退化为标准 FlashAttention）
    {"batch_size": 1, "num_heads": 1, "seq_len": 64, "head_dim": 64, "l": 1, "direction": 0},
    
    # 最大区域数量（l=S）
    {"batch_size": 1, "num_heads": 1, "seq_len": 64, "head_dim": 64, "l": 64, "direction": 0},
    
    # 最小 head_dim（需要对齐到 16）
    {"batch_size": 1, "num_heads": 1, "seq_len": 64, "head_dim": 16, "l": 4, "direction": 0},
    
    # 最大 head_dim
    {"batch_size": 1, "num_heads": 1, "seq_len": 64, "head_dim": 256, "l": 4, "direction": 0},
    
    # 单头注意力
    {"batch_size": 1, "num_heads": 1, "seq_len": 256, "head_dim": 64, "l": 4, "direction": 0},
    
    # 大批量单头
    {"batch_size": 32, "num_heads": 1, "seq_len": 128, "head_dim": 64, "l": 4, "direction": 0},
    
    # 不同 scale 值
    {"batch_size": 1, "num_heads": 1, "seq_len": 64, "head_dim": 64, "l": 4, "direction": 0, "scale": 0.125},
    {"batch_size": 1, "num_heads": 1, "seq_len": 64, "head_dim": 64, "l": 4, "direction": 0, "scale": 1.0},
    {"batch_size": 1, "num_heads": 1, "seq_len": 64, "head_dim": 64, "l": 4, "direction": 0, "scale": 2.0},
]
```

## 5. 算子标杆

### 5.1 CPU 参考实现

```python
import torch
import torch.nn.functional as F
import math

def area_flash_attention_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    l: int = 4,
    direction: int = 0,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Area Flash Attention CPU 参考实现
    
    Args:
        query: [B, N, S, D]
        key: [B, N, S, D]
        value: [B, N, S, D]
        l: 区域数量
        direction: 切分方向（0=垂直，1=水平）
        scale: 缩放因子
    
    Returns:
        output: [B, N, S, D]
    """
    B, N, S, D = query.shape
    
    # 验证参数
    assert S % l == 0, f"seq_len {S} must be divisible by l {l}"
    
    # 计算每个区域的序列长度
    S_region = S // l
    
    # 初始化输出
    output = torch.zeros_like(query)
    
    # 对每个区域独立计算注意力
    for region_idx in range(l):
        # 计算区域的起始和结束位置
        start_idx = region_idx * S_region
        end_idx = (region_idx + 1) * S_region
        
        # 提取区域的 Q、K、V
        q_region = query[:, :, start_idx:end_idx, :]  # [B, N, S_region, D]
        k_region = key[:, :, start_idx:end_idx, :]    # [B, N, S_region, D]
        v_region = value[:, :, start_idx:end_idx, :]  # [B, N, S_region, D]
        
        # 计算注意力分数
        attn_scores = torch.matmul(q_region, k_region.transpose(-2, -1))  # [B, N, S_region, S_region]
        attn_scores = attn_scores * scale
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 计算输出
        region_output = torch.matmul(attn_weights, v_region)  # [B, N, S_region, D]
        
        # 写回输出
        output[:, :, start_idx:end_idx, :] = region_output
    
    return output
```

### 5.2 NPU 调用方式

```python
import torch
import torch_npu

def area_flash_attention_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    l: int = 4,
    direction: int = 0,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Area Flash Attention NPU 实现
    
    Args:
        query: [B, N, S, D] - 必须在 NPU 上
        key: [B, N, S, D] - 必须在 NPU 上
        value: [B, N, S, D] - 必须在 NPU 上
        l: 区域数量
        direction: 切分方向（0=垂直，1=水平）
        scale: 缩放因子
    
    Returns:
        output: [B, N, S, D] - 在 NPU 上
    """
    # 确保输入在 NPU 上
    assert query.device.type == 'npu', "query must be on NPU"
    assert key.device.type == 'npu', "key must be on NPU"
    assert value.device.type == 'npu', "value must be on NPU"
    
    # 调用自定义算子
    output = torch.ops.ops_transformer.area_flash_attention(
        query, key, value, l, direction, scale
    )
    
    return output
```

### 5.3 精度对比函数

```python
def compare_precision(
    output_npu: torch.Tensor,
    output_cpu: torch.Tensor,
    dtype: torch.dtype,
) -> dict:
    """
    对比 NPU 和 CPU 输出的精度
    
    Args:
        output_npu: NPU 输出
        output_cpu: CPU 输出
        dtype: 数据类型
    
    Returns:
        metrics: 精度指标字典
    """
    # 将 NPU 输出移到 CPU
    output_npu_cpu = output_npu.cpu()
    
    # 计算精度指标
    max_abs_error = torch.max(torch.abs(output_npu_cpu - output_cpu)).item()
    mean_abs_error = torch.mean(torch.abs(output_npu_cpu - output_cpu)).item()
    
    # 避免除零
    rel_error = torch.abs(output_npu_cpu - output_cpu) / (torch.abs(output_cpu) + 1e-8)
    max_rel_error = torch.max(rel_error).item()
    mean_rel_error = torch.mean(rel_error).item()
    
    # 余弦相似度
    cosine_sim = F.cosine_similarity(
        output_npu_cpu.flatten(),
        output_cpu.flatten(),
        dim=0
    ).item()
    
    # 根据数据类型设置阈值
    if dtype == torch.float16:
        threshold_abs = 1e-3
        threshold_rel = 1e-2
    elif dtype == torch.bfloat16:
        threshold_abs = 1e-2
        threshold_rel = 1e-1
    else:  # float32
        threshold_abs = 1e-5
        threshold_rel = 1e-4
    
    # 判断是否通过
    passed = (
        max_abs_error < threshold_abs and
        cosine_sim > 0.999
    )
    
    return {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "max_rel_error": max_rel_error,
        "mean_rel_error": mean_rel_error,
        "cosine_sim": cosine_sim,
        "passed": passed,
        "threshold_abs": threshold_abs,
        "threshold_rel": threshold_rel,
    }
```

## 6. 测试用例统计

- **TEST_SHAPES**: 13 个常规测试用例
- **GENERAL_SHAPES**: 10 个泛化测试用例
- **BOUNDARY_VALUES**: 10 个边界值测试用例
- **总计**: 33 个测试用例

每个测试用例会在 3 种数据类型（float16, float32, bfloat16）下运行，总计 **99 个测试场景**。

## 7. 测试覆盖率

### 7.1 Shape 覆盖
- ✓ 不同 batch_size（1, 2, 4, 8, 16, 32）
- ✓ 不同 num_heads（1, 2, 4, 8, 16, 32）
- ✓ 不同 seq_len（4, 32, 64, 128, 256, 512, 784, 1024, 196, 3136, 2048）
- ✓ 不同 head_dim（16, 32, 64, 128, 256）

### 7.2 参数覆盖
- ✓ 不同区域数量 l（1, 2, 4, 8, 16, 64）
- ✓ 不同切分方向（0=垂直, 1=水平）
- ✓ 不同缩放因子（0.125, 1.0, 2.0）

### 7.3 数据类型覆盖
- ✓ float16
- ✓ float32
- ✓ bfloat16

## 8. 测试执行命令

```bash
# 运行所有测试
pytest tests/test_area_flash_attention.py -v

# 运行特定数据类型的测试
pytest tests/test_area_flash_attention.py -v -k "float16"

# 运行精度测试
pytest tests/test_area_flash_attention_precision.py -v

# 生成精度报告
python tests/run_area_flash_attention_precision_report.py

# 生成性能报告
python tests/benchmark_area_flash_attention_msprof.py
```
