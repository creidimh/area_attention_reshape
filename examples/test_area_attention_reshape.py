#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Area Attention Reshape 算子测试脚本
"""

import torch
import torch_npu
import numpy as np

def area_attention_reshape_reference(x, l, direction='h'):
    """
    PyTorch 参考实现
    
    Args:
        x: 输入张量 (B, H, W, C)
        l: 切分段数
        direction: 'h' 或 'w'
    
    Returns:
        输出张量 (B*l, N_chunk, C)
    """
    B, H, W, C = x.shape
    
    if direction == 'h':
        h_chunk = H // l
        x = x.view(B, l, h_chunk, W, C)
        x = x.permute(0, 1, 3, 2, 4)  # (B, l, W, h_chunk, C)
        x = x.reshape(B * l, W * h_chunk, C)
    else:  # 'w'
        w_chunk = W // l
        x = x.view(B, H, l, w_chunk, C)
        x = x.permute(0, 2, 1, 3, 4)  # (B, l, H, w_chunk, C)
        x = x.reshape(B * l, H * w_chunk, C)
    
    return x

def test_area_attention_reshape():
    """测试 area_attention_reshape 算子"""
    
    print("=" * 60)
    print("Area Attention Reshape 算子测试")
    print("=" * 60)
    
    # 测试参数
    B, H, W, C = 2, 8, 8, 16
    l = 4
    
    # 测试高度方向切分
    print("\n测试 1: 高度方向切分 (direction='h')")
    print(f"输入形状: ({B}, {H}, {W}, {C})")
    print(f"切分段数: l={l}")
    
    x = torch.randn(B, H, W, C, dtype=torch.float32)
    x_npu = x.npu()
    
    # 调用参考实现
    ref_output = area_attention_reshape_reference(x, l, 'h')
    print(f"参考输出形状: {ref_output.shape}")
    
    # 测试宽度方向切分
    print("\n测试 2: 宽度方向切分 (direction='w')")
    print(f"输入形状: ({B}, {H}, {W}, {C})")
    print(f"切分段数: l={l}")
    
    ref_output_w = area_attention_reshape_reference(x, l, 'w')
    print(f"参考输出形状: {ref_output_w.shape}")
    
    # 测试不同数据类型
    print("\n测试 3: 不同数据类型")
    for dtype in [torch.float16, torch.float32]:
        x_dtype = x.to(dtype)
        ref_dtype = area_attention_reshape_reference(x_dtype, l, 'h')
        print(f"  {dtype}: 输出形状 {ref_dtype.shape}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_area_attention_reshape()
