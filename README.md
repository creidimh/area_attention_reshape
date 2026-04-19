# AreaAttentionReshape

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|

## 功能说明

- 算子功能：实现 YOLOv12 Area Attention 的条带重塑操作，将特征图沿高度或宽度方向均匀切分为若干条带，通过 reshape 和维度置换实现高效的条带内注意力计算。

- 计算公式：

**高度方向切分（direction='h'）：**
```
输入：x (B, H, W, C)
参数：l (切分段数), direction='h'

步骤：
1. h_chunk = H // l
2. x = x.view(B, l, h_chunk, W, C)
3. x = x.permute(0, 1, 3, 2, 4)      # (B, l, W, h_chunk, C)
4. x = x.reshape(B*l, W*h_chunk, C)  # 合并条带与batch

输出：(B*l, W*h_chunk, C)
```

**宽度方向切分（direction='w'）：**
```
输入：x (B, H, W, C)
参数：l (切分段数), direction='w'

步骤：
1. w_chunk = W // l
2. x = x.view(B, H, l, w_chunk, C)
3. x = x.permute(0, 2, 1, 3, 4)      # (B, l, H, w_chunk, C)
4. x = x.reshape(B*l, H*w_chunk, C)  # 合并条带与batch

输出：(B*l, H*w_chunk, C)
```

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>输入特征图，形状为 (B, H, W, C)，其中 B 为 batch size，H、W 为空间尺寸，C 为通道数。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>重塑后的特征图，形状为 (B*l, N_chunk, C)，其中 N_chunk = h_chunk*W（高度切分）或 H*w_chunk（宽度切分）。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l</td>
      <td>属性</td>
      <td>切分段数，默认值为 4。特征图的高度或宽度必须能被 l 整除。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>direction</td>
      <td>属性</td>
      <td>切分方向，'h' 表示沿高度方向切分，'w' 表示沿宽度方向切分。默认值为 'h'。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

1. 输入特征图的高度（direction='h'）或宽度（direction='w'）必须能被切分段数 `l` 整除。
2. YOLO 骨干网络输出的特征图尺寸通常为 80×80、40×40、20×20 等，均能被 4 整除，满足默认切分段数要求。
3. 切分段数 `l` 的取值范围为 [1, H]（高度切分）或 [1, W]（宽度切分）。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn调用</td>
    <td><a href="./examples/test_aclnn_area_attention_reshape.cpp">test_aclnn_area_attention_reshape</a></td>
    <td rowspan="2">参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>

## 算子实现原理

Area Attention 通过将特征图切分为多个条带，在每个条带内部独立计算自注意力，从而将计算复杂度从 O(n²) 降低至 O(n²/l)（当 l=4 时为 O(n²/4)），且无需显式的窗口分区与反转操作，仅通过 reshape 和维度置换即可高效实现。

该算子是 Area Attention 模块的核心组件之一，用于实现特征图的条带切分和重塑操作。
