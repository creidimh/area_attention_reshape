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
 * \file area_attention_reshape.cpp
 * \brief Area Attention Reshape Kernel Entry
 */

#include "area_attention_reshape.h"

enum class AreaAttentionReshapeTilingKey : uint32_t
{
    TILING_KEY_FLOAT16 = 0,
    TILING_KEY_FLOAT = 1,
    TILING_KEY_BF16 = 2,
};

template <uint32_t schMode>
__global__ __aicore__ void area_attention_reshape(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AreaAttentionReshapeTilingData);
    GET_TILING_DATA_WITH_STRUCT(AreaAttentionReshapeTilingData, tilingData, tiling);

    if constexpr (schMode == static_cast<uint32_t>(AreaAttentionReshapeTilingKey::TILING_KEY_FLOAT16)) {
        NsAreaAttentionReshape::AreaAttentionReshape<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }

    if constexpr (schMode == static_cast<uint32_t>(AreaAttentionReshapeTilingKey::TILING_KEY_FLOAT)) {
        NsAreaAttentionReshape::AreaAttentionReshape<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }

    if constexpr (schMode == static_cast<uint32_t>(AreaAttentionReshapeTilingKey::TILING_KEY_BF16)) {
        NsAreaAttentionReshape::AreaAttentionReshape<bfloat16_t> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
