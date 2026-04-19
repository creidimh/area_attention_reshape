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
 * \file area_attention_reshape_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeAreaAttentionReshape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeAreaAttentionReshape");

    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    
    int64_t l = *attrs->GetInt(0);
    int64_t direction = attrs->GetInt(1) != nullptr ? *attrs->GetInt(1) : 0;

    int64_t batchSize = xShape->GetDim(0);
    int64_t height = xShape->GetDim(1);
    int64_t width = xShape->GetDim(2);
    int64_t channels = xShape->GetDim(3);

    int64_t newBatchSize = batchSize * l;
    int64_t nChunk;
    if (direction == 0) {
        nChunk = width * (height / l);
    } else {
        nChunk = height * (width / l);
    }

    yShape->SetDimNum(3);
    yShape->SetDim(0, newBatchSize);
    yShape->SetDim(1, nChunk);
    yShape->SetDim(2, channels);

    OP_LOGD(context->GetNodeName(), "End to do InferShapeAreaAttentionReshape");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AreaAttentionReshape).InferShape(InferShapeAreaAttentionReshape);
} // namespace ops