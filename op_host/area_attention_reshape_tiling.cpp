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
 * \file area_attention_reshape_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../op_kernel/area_attention_reshape_tiling_data.h"
#include "../op_kernel/area_attention_reshape_tiling_key.h"

namespace optiling {

using namespace Ops::Transformer::OpTiling;

const uint32_t BLOCK_DIM = 8;
const int64_t TILE_NUM = 8;
const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
const int32_t DIMS_LIMIT = 4;
constexpr int32_t ATTRPOS0 = 0;
constexpr uint32_t INDEXZERO = 0;
constexpr uint32_t INDEXONE = 1;
constexpr uint32_t INDEXTWO = 2;
constexpr uint32_t INDEXTHREE = 3;
constexpr int64_t DIRECTION_H = 0;
constexpr int64_t DIRECTION_W = 1;

struct AreaAttentionReshapeCompileInfo {};

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    // 获取ubsize coreNum
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取属性，shape信息
ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalElements, ge::DataType& dataType,
                                      int64_t& batchSize, int64_t& height, int64_t& width, int64_t& channels,
                                      int64_t& l, int64_t& direction)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());

    OP_CHECK_IF(
        inputShapeX.GetDimNum() != DIMS_LIMIT,
        OP_LOGE(context, "AreaAttentionReshape: input shape dim = %zu, should be 4", inputShapeX.GetDimNum()),
        return ge::GRAPH_FAILED);

    batchSize = inputShapeX.GetDim(INDEXZERO);
    height = inputShapeX.GetDim(INDEXONE);
    width = inputShapeX.GetDim(INDEXTWO);
    channels = inputShapeX.GetDim(INDEXTHREE);
    totalElements = batchSize * height * width * channels;

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "AreaAttentionReshape: unsupported dtype");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* lPtr = attrs->GetInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, lPtr);
    l = *lPtr;
    const int64_t* directionPtr = attrs->GetInt(1);
    direction = directionPtr != nullptr ? *directionPtr : DIRECTION_H;

    if (direction == DIRECTION_H) {
        OP_CHECK_IF(
            height % l != 0,
            OP_LOGE(context, "AreaAttentionReshape: height %ld must be divisible by l %ld", height, l),
            return ge::GRAPH_FAILED);
    } else if (direction == DIRECTION_W) {
        OP_CHECK_IF(
            width % l != 0,
            OP_LOGE(context, "AreaAttentionReshape: width %ld must be divisible by l %ld", width, l),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(context, "AreaAttentionReshape: invalid direction %ld, must be 0 (h) or 1 (w)", direction);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus AreaAttentionReshapeTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    int64_t totalElements;
    ge::DataType dataType;
    int64_t batchSize, height, width, channels, l, direction;

    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalElements, dataType, batchSize, height, width, channels, l, direction) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    AreaAttentionReshapeTilingData* tiling = context->GetTilingData<AreaAttentionReshapeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(AreaAttentionReshapeTilingData), 0, sizeof(AreaAttentionReshapeTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->batchSize = batchSize;
    tiling->height = height;
    tiling->width = width;
    tiling->channels = channels;
    tiling->l = l;
    tiling->direction = direction;
    tiling->totalElements = totalElements;
    tiling->tileNum = TILE_NUM;

    context->SetBlockDim(BLOCK_DIM);
    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
        context->SetTilingKey(tilingKey);
    } else if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
        context->SetTilingKey(tilingKey);
    } else if (dataType == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_2);
        context->SetTilingKey(tilingKey);
    } else {
        OP_LOGE(context, "get dtype error");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAreaAttentionReshape([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(AreaAttentionReshape).Tiling(AreaAttentionReshapeTilingFunc).TilingParse<AreaAttentionReshapeCompileInfo>(TilingParseForAreaAttentionReshape);
} // namespace optiling
