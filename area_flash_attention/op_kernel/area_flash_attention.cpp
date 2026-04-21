/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file area_flash_attention.cpp
 * \brief AreaFlashAttention Kernel 入口
 */

#include "area_flash_attention.h"

enum class AreaFlashAttentionTilingKey : uint32_t
{
    TILING_KEY_FLOAT16 = 0,
    TILING_KEY_FLOAT = 1,
    TILING_KEY_BF16 = 2,
};

template <uint32_t schMode>
__global__ __aicore__ void area_flash_attention(
    GM_ADDR query,
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR attentionOut,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AreaFlashAttentionTilingData);
    GET_TILING_DATA_WITH_STRUCT(AreaFlashAttentionTilingData, tilingData, tiling);

    if constexpr (schMode == static_cast<uint32_t>(AreaFlashAttentionTilingKey::TILING_KEY_FLOAT16)) {
        NsAreaFlashAttention::AreaFlashAttention<half> op;
        op.Init(query, key, value, attentionOut, &tilingData);
        op.Process();
    }

    if constexpr (schMode == static_cast<uint32_t>(AreaFlashAttentionTilingKey::TILING_KEY_FLOAT)) {
        NsAreaFlashAttention::AreaFlashAttention<float> op;
        op.Init(query, key, value, attentionOut, &tilingData);
        op.Process();
    }

    if constexpr (schMode == static_cast<uint32_t>(AreaFlashAttentionTilingKey::TILING_KEY_BF16)) {
        NsAreaFlashAttention::AreaFlashAttention<bfloat16_t> op;
        op.Init(query, key, value, attentionOut, &tilingData);
        op.Process();
    }
}
