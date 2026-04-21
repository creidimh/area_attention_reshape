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
 * \file area_flash_attention_infershape.cpp
 * \brief AreaFlashAttention InferShape 实现
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;
static constexpr int64_t IDX_2 = 2;

static ge::graphStatus InferShapeAreaFlashAttention(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeAreaFlashAttention");

    const gert::Shape* queryShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    const gert::Shape* keyShape = context->GetInputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);

    const gert::Shape* valueShape = context->GetInputShape(IDX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);

    gert::Shape* outShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    auto queryShapeSize = queryShape->GetDimNum();
    OP_CHECK_IF(queryShapeSize != 4,
        OP_LOGE(context, "query shape dim = %zu, should be 4", queryShapeSize),
        return GRAPH_FAILED);

    auto keyShapeSize = keyShape->GetDimNum();
    OP_CHECK_IF(keyShapeSize != 4,
        OP_LOGE(context, "key shape dim = %zu, should be 4", keyShapeSize),
        return GRAPH_FAILED);

    auto valueShapeSize = valueShape->GetDimNum();
    OP_CHECK_IF(valueShapeSize != 4,
        OP_LOGE(context, "value shape dim = %zu, should be 4", valueShapeSize),
        return GRAPH_FAILED);

    for (size_t i = 0; i < queryShapeSize; i++) {
        int64_t queryDim = queryShape->GetDim(i);
        int64_t keyDim = keyShape->GetDim(i);
        int64_t valueDim = valueShape->GetDim(i);

        OP_CHECK_IF(queryDim != keyDim,
            OP_LOGE(context, "query and key shape mismatch at dim %zu: %ld vs %ld", i, queryDim, keyDim),
            return GRAPH_FAILED);

        OP_CHECK_IF(queryDim != valueDim,
            OP_LOGE(context, "query and value shape mismatch at dim %zu: %ld vs %ld", i, queryDim, valueDim),
            return GRAPH_FAILED);
    }

    outShape->SetDimNum(queryShapeSize);
    for (size_t i = 0; i < queryShapeSize; i++) {
        int64_t dim = queryShape->GetDim(i);
        outShape->SetDim(i, dim);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeAreaFlashAttention");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AreaFlashAttention).InferShape(InferShapeAreaFlashAttention);
} // namespace ops
