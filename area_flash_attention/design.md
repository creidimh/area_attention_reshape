# AreaFlashAttention 算子设计文档

## 1. 算子概述

### 1.1 算子名称
`AreaFlashAttention`

### 1.2 功能描述
AreaFlashAttention 是一个融合算子，将 YOLOv12 中的 Area Attention 机制与 FlashAttention 技术有机结合：

- **Area Attention**：将特征图沿垂直或水平方向切分为 `l` 个独立区域，每个区域内独立计算注意力，将复杂度从 `O(n²)` 降低到 `O(n²/l)`
- **FlashAttention**：通过分块计算和在线 softmax 算法，避免物化完整的注意力矩阵，大幅减少 HBM 访问

### 1.3 数学公式

对于输入特征图 `[B, H, W, C]`，切分为 `l` 个区域：

**垂直切分（direction=0）**：
- 每个区域大小：`[B, H/l, W, C]`
- 区域序列长度：`S_region = (H/l) × W`

**水平切分（direction=1）**：
- 每个区域大小：`[B, H, W/l, C]`
- 区域序列长度：`S_region = H × (W/l)`

**区域内 FlashAttention 计算**：
```
对于每个区域 r ∈ [0, l):
    Q_r, K_r, V_r = split_by_region(Q, K, V, r)
    
    # FlashAttention 分块计算
    O_r = FlashAttention(Q_r, K_r, V_r)
    
    # 在线 softmax 更新
    for i in range(0, S_region, B_r):
        for j in range(0, S_region, B_c):
            S_ij = Q_r[i:i+B_r] @ K_r[j:j+B_c].T
            m_new = max(m_i, row_max(S_ij))
            P_ij = exp(S_ij - m_new)
            O_i = scale * O_i + P_ij @ V_r[j:j+B_c]
```

## 2. 接口定义

### 2.1 函数签名

```cpp
// C++ 接口
void area_flash_attention(
    const Tensor& query,      // [B, N, S, D] 或 [B, H, W, C]
    const Tensor& key,        // [B, N, S, D] 或 [B, H, W, C]
    const Tensor& value,      // [B, N, S, D] 或 [B, H, W, C]
    Tensor& attention_out,    // [B, N, S, D] 或 [B, H, W, C]
    int64_t l = 4,            // 区域数量，默认4
    int64_t direction = 0,    // 切分方向：0=垂直，1=水平
    double scale = 1.0,       // 缩放因子
    int64_t layout = 0        // 布局：0=BSND, 1=BNSD
);
```

### 2.2 Python 接口

```python
def area_flash_attention(
    query: torch.Tensor,      # [B, N, S, D] 或 [B, H, W, C]
    key: torch.Tensor,        # [B, N, S, D] 或 [B, H, W, C]
    value: torch.Tensor,      # [B, N, S, D] 或 [B, H, W, C]
    l: int = 4,               # 区域数量
    direction: int = 0,       # 切分方向：0=垂直，1=水平
    scale: float = 1.0,       # 缩放因子
    layout: int = 0           # 布局：0=BSND, 1=BNSD
) -> torch.Tensor:
    """
    Area Flash Attention 融合算子
    
    Args:
        query: Query 张量，形状为 [B, N, S, D] 或 [B, H, W, C]
        key: Key 张量，形状与 query 相同
        value: Value 张量，形状与 query 相同
        l: 区域切分数量，默认为 4
        direction: 切分方向，0 表示垂直切分，1 表示水平切分
        scale: 注意力分数的缩放因子，默认为 1.0
        layout: 输入布局，0 表示 BSND，1 表示 BNSD
    
    Returns:
        attention_out: 注意力输出，形状与 query 相同
    """
```

## 3. 支持的数据类型

| 数据类型 | 支持状态 | 说明 |
|---------|---------|------|
| float16 | ✓ | 推荐，性能最优 |
| float32 | ✓ | 高精度场景 |
| bfloat16 | ✓ | 训练场景 |

## 4. Tiling 策略

### 4.1 两层 Tiling 设计

本算子采用两层 Tiling 策略：

#### 第一层：区域级切分（Region Tiling）
- 沿高度或宽度方向将特征图切分为 `l` 个独立区域
- 每个区域分配给独立的 AI Core 处理
- 区域之间无数据依赖，天然支持多核并行

#### 第二层：块级切分（Block Tiling）
- 每个区域内部，应用 FlashAttention 的分块策略
- 将 Q、K、V 矩阵沿序列维度切分为基本块
- 基本块大小需考虑 UB 容量限制

### 4.2 Tiling 参数

| 参数 | 说明 | 计算方式 |
|------|------|----------|
| `block_dim` | 使用的核数 | `min(l * batch_size, coreNum)` |
| `Br` | Q 块的行数 | 根据 UB 容量计算，建议 64-128 |
| `Bc` | K/V 块的列数 | 根据 UB 容量计算，建议 64-128 |
| `tile_num` | 每个核内的分块数 | `ceil(S_region / (Br * Bc))` |

### 4.3 UB 容量计算

假设 UB 大小为 `ubSize`，数据类型为 `T`，每个元素大小为 `sizeof(T)`：

```
UB 需求 = Br * D * sizeof(T)        // Q 块
        + Bc * D * sizeof(T)        // K 块
        + Bc * D * sizeof(T)        // V 块
        + Br * Bc * sizeof(float)   // 注意力分数（float32）
        + Br * D * sizeof(T)        // 输出块
        + Br * sizeof(float)        // softmax max
        + Br * sizeof(float)        // softmax sum
```

约束条件：
```
UB 需求 ≤ ubSize * 0.9  // 预留 10% 安全余量
```

## 5. 计算逻辑伪代码

### 5.1 Host 端 Tiling 计算

```cpp
ge::graphStatus AreaFlashAttentionTiling(gert::TilingContext* context) {
    // 1. 获取平台信息
    uint64_t ubSize;
    int64_t coreNum;
    GetPlatformInfo(context, ubSize, coreNum);
    
    // 2. 获取输入 shape 和属性
    auto queryShape = context->GetInputShape(0)->GetStorageShape();
    int64_t B = queryShape.GetDim(0);  // batch size
    int64_t N = queryShape.GetDim(1);  // num heads
    int64_t S = queryShape.GetDim(2);  // sequence length
    int64_t D = queryShape.GetDim(3);  // head dimension
    
    int64_t l = context->GetAttrs()->GetInt(0);      // 区域数量
    int64_t direction = context->GetAttrs()->GetInt(1);  // 切分方向
    
    // 3. 计算区域大小
    int64_t S_region = S / l;  // 每个区域的序列长度
    
    // 4. 计算最优分块大小
    int64_t Br, Bc;
    CalculateOptimalBlockSize(ubSize, D, sizeof(T), Br, Bc);
    
    // 5. 设置 Tiling 数据
    tiling->batchSize = B;
    tiling->numHeads = N;
    tiling->seqLength = S;
    tiling->headDim = D;
    tiling->regionNum = l;
    tiling->direction = direction;
    tiling->blockBr = Br;
    tiling->blockBc = Bc;
    tiling->seqLengthPerRegion = S_region;
    
    // 6. 设置核数
    int64_t totalRegions = B * N * l;
    context->SetBlockDim(min(totalRegions, coreNum));
    
    return ge::GRAPH_SUCCESS;
}
```

### 5.2 Kernel 端计算流程

```cpp
template<typename T>
__aicore__ void AreaFlashAttention<T>::Process() {
    // 1. 获取当前核处理的区域信息
    int64_t blockIdx = GetBlockIdx();
    int64_t batchIdx = blockIdx / (numHeads_ * regionNum_);
    int64_t headIdx = (blockIdx / regionNum_) % numHeads_;
    int64_t regionIdx = blockIdx % regionNum_;
    
    // 2. 计算区域的 Q、K、V 起始地址
    int64_t regionOffset = CalculateRegionOffset(batchIdx, headIdx, regionIdx);
    GlobalTensor<T> queryRegion = queryGM[regionOffset];
    GlobalTensor<T> keyRegion = keyGM[regionOffset];
    GlobalTensor<T> valueRegion = valueGM[regionOffset];
    
    // 3. FlashAttention 分块计算
    for (int64_t i = 0; i < seqLengthPerRegion_; i += blockBr_) {
        // 3.1 加载 Q 块
        LocalTensor<T> qBlock = qQueue.AllocTensor<T>();
        DataCopy(qBlock, queryRegion[i * headDim_], blockBr_ * headDim_);
        qQueue.EnQue(qBlock);
        
        // 3.2 初始化输出和 softmax 状态
        LocalTensor<T> oBlock = oQueue.AllocTensor<T>();
        LocalTensor<float> maxLocal = maxQueue.AllocTensor<float>();
        LocalTensor<float> sumLocal = sumQueue.AllocTensor<float>();
        
        Duplicate(maxLocal, -INFINITY, blockBr_);
        Duplicate(sumLocal, 0.0f, blockBr_);
        Duplicate(oBlock, 0, blockBr_ * headDim_);
        
        // 3.3 遍历 K、V 块
        for (int64_t j = 0; j < seqLengthPerRegion_; j += blockBc_) {
            // 加载 K、V 块
            LocalTensor<T> kBlock = kQueue.AllocTensor<T>();
            LocalTensor<T> vBlock = vQueue.AllocTensor<T>();
            DataCopy(kBlock, keyRegion[j * headDim_], blockBc_ * headDim_);
            DataCopy(vBlock, valueRegion[j * headDim_], blockBc_ * headDim_);
            
            // 计算 QK^T
            LocalTensor<float> attnScores = attnQueue.AllocTensor<float>();
            MatMul(attnScores, qBlock, kBlock, blockBr_, blockBc_, headDim_);
            
            // 缩放
            Muls(attnScores, attnScores, scale_, blockBr_ * blockBc_);
            
            // 在线 softmax 更新
            LocalTensor<float> maxNew = tempQueue.AllocTensor<float>();
            LocalTensor<float> sumNew = tempQueue.AllocTensor<float>();
            
            // 计算新的最大值
            ReduceMax(maxNew, attnScores, blockBr_);
            Max(maxNew, maxLocal, maxNew, blockBr_);
            
            // 计算缩放因子
            LocalTensor<float> scaleOld = tempQueue.AllocTensor<float>();
            Sub(scaleOld, maxLocal, maxNew, blockBr_);
            Exp(scaleOld, scaleOld, blockBr_);
            
            // 更新 sum
            Sub(attnScores, attnScores, maxNew, blockBr_ * blockBc_);
            Exp(attnScores, attnScores, blockBr_ * blockBc_);
            ReduceSum(sumNew, attnScores, blockBr_);
            Mul(sumNew, scaleOld, sumNew, blockBr_);
            Add(sumNew, sumLocal, sumNew, blockBr_);
            
            // 更新输出
            Mul(oBlock, oBlock, scaleOld, blockBr_ * headDim_);
            MatMul(oBlock, attnScores, vBlock, blockBr_, headDim_, blockBc_, true);
            
            // 更新状态
            Copy(maxLocal, maxNew, blockBr_);
            Copy(sumLocal, sumNew, blockBr_);
            
            // 释放临时 buffer
            kQueue.FreeTensor(kBlock);
            vQueue.FreeTensor(vBlock);
            attnQueue.FreeTensor(attnScores);
            tempQueue.FreeTensor(maxNew);
            tempQueue.FreeTensor(sumNew);
            tempQueue.FreeTensor(scaleOld);
        }
        
        // 3.4 最终归一化
        LocalTensor<float> invSum = tempQueue.AllocTensor<float>();
        Rec(invSum, sumLocal, blockBr_);
        Mul(oBlock, oBlock, invSum, blockBr_ * headDim_);
        
        // 3.5 写回输出
        DataCopy(outputGM[regionOffset + i * headDim_], oBlock, blockBr_ * headDim_);
        
        // 释放 buffer
        qQueue.FreeTensor(qBlock);
        oQueue.FreeTensor(oBlock);
        maxQueue.FreeTensor(maxLocal);
        sumQueue.FreeTensor(sumLocal);
        tempQueue.FreeTensor(invSum);
    }
}
```

## 6. UB 分配表

### 6.1 Buffer 列表

| Buffer 名称 | 用途 | 大小计算 | 数据类型 |
|------------|------|----------|----------|
| `qQueue` | Q 块缓存 | `Br * D` | T |
| `kQueue` | K 块缓存 | `Bc * D` | T |
| `vQueue` | V 块缓存 | `Bc * D` | T |
| `oQueue` | 输出块缓存 | `Br * D` | T |
| `attnQueue` | 注意力分数 | `Br * Bc` | float |
| `maxQueue` | softmax 最大值 | `Br` | float |
| `sumQueue` | softmax 求和 | `Br` | float |
| `tempQueue` | 临时计算 | `max(Br, Br*D)` | float |

### 6.2 UB 总需求

```
UB_total = Br * D * sizeof(T)          // qQueue
         + Bc * D * sizeof(T)          // kQueue
         + Bc * D * sizeof(T)          // vQueue
         + Br * D * sizeof(T)          // oQueue
         + Br * Bc * sizeof(float)     // attnQueue
         + Br * sizeof(float)          // maxQueue
         + Br * sizeof(float)          // sumQueue
         + max(Br, Br*D) * sizeof(float)  // tempQueue
```

### 6.3 bufferCoefficient

| 数据类型 | bufferCoefficient |
|---------|-------------------|
| float16 | 2.0 |
| float32 | 4.0 |
| bfloat16 | 2.0 |

计算示例（float16，Br=128, Bc=128, D=64）：
```
UB_total = 128*64*2 + 128*64*2 + 128*64*2 + 128*64*2
         + 128*128*4 + 128*4 + 128*4 + 128*64*4
         = 65536 + 65536 + 65536 + 65536 + 65536 + 512 + 512 + 32768
         = 361472 bytes ≈ 353 KB
```

## 7. 约束条件

### 7.1 Shape 约束
- 输入 Q、K、V 的 shape 必须相同
- 序列长度 S 必须能被区域数量 l 整除
- 头维度 D 建议为 16 的倍数（对齐 Cube 计算单元）

### 7.2 数据类型约束
- Q、K、V 的数据类型必须相同
- 支持的数据类型：float16, float32, bfloat16

### 7.3 参数约束
- `l` ≥ 1 且 `l` ≤ S
- `direction` ∈ {0, 1}
- `scale` > 0

## 8. 性能优化策略

### 8.1 多核并行
- 每个区域分配给独立的 AI Core
- 区域之间无数据依赖，实现线性加速

### 8.2 双缓冲流水
- 使用 ping-pong buffer 实现计算与搬运并行
- 利用 Ascend C 的 `TPipe` 进行流水编排

### 8.3 Cube 与 Vector 协同
- 矩阵乘法（QK^T, PV）使用 Cube 单元
- Softmax 计算（exp, reduce）使用 Vector 单元
- 通过流水掩盖延迟

### 8.4 内存访问优化
- 连续访问 Global Memory
- 合理设置分块大小以最大化 UB 利用率
- 避免频繁的小数据块搬运

## 9. 验证方法

### 9.1 功能验证
- 与 PyTorch 标准 Attention 实现对比
- 验证不同区域数量和切分方向的正确性

### 9.2 精度验证
- 最大绝对误差 < 1e-3（float16）
- 最大相对误差 < 1e-2（float16）
- 余弦相似度 > 0.999

### 9.3 性能验证
- 使用 msprof 工具采集性能数据
- 对比标准 Attention 的性能提升
- 分析内存带宽利用率
