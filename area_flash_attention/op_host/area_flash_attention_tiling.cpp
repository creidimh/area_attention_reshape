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
 * \file area_flash_attention_tiling.cpp
 * \brief AreaFlashAttention Tiling 实现
 */

#include "log/log.h"
#include "util/math_util.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../op_kernel/area_flash_attention_tiling_data.h"
#include "../op_kernel/area_flash_attention_tiling_key.h"
#include <cmath>

namespace optiling {

using namespace Ops::Transformer::OpTiling;

const uint32_t BLOCK_DIM = 8;
const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
const int32_t DIMS_LIMIT = 4;
constexpr int32_t ATTRPOS0 = 0;
constexpr uint32_t INDEXZERO = 0;
constexpr uint32_t INDEXONE = 1;
constexpr uint32_t INDEXTWO = 2;
constexpr uint32_t INDEXTHREE = 3;
constexpr int64_t DIRECTION_H = 0;
constexpr int64_t DIRECTION_W = 1;
constexpr int64_t ALIGNMENT = 16;
constexpr double UB_USAGE_RATIO = 0.9;

struct AreaFlashAttentionCompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context,
    int64_t& batchSize,
    int64_t& numHeads,
    int64_t& seqLength,
    int64_t& headDim,
    int64_t& l,
    int64_t& direction,
    double& scale,
    int64_t& layout,
    ge::DataType& dataType)
{
    auto queryShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    auto inputShape = EnsureNotScalar(queryShape->GetStorageShape());

    OP_CHECK_IF(
        inputShape.GetDimNum() != DIMS_LIMIT,
        OP_LOGE(context, "AreaFlashAttention: input shape dim = %zu, should be 4", inputShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    batchSize = inputShape.GetDim(INDEXZERO);
    numHeads = inputShape.GetDim(INDEXONE);
    seqLength = inputShape.GetDim(INDEXTWO);
    headDim = inputShape.GetDim(INDEXTHREE);

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "AreaFlashAttention: unsupported dtype");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* lPtr = attrs->GetInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, lPtr);
    l = *lPtr;

    const int64_t* directionPtr = attrs->GetInt(1);
    direction = directionPtr != nullptr ? *directionPtr : DIRECTION_H;

    const double* scalePtr = attrs->GetFloat(2);
    scale = scalePtr != nullptr ? *scalePtr : 1.0;

    const int64_t* layoutPtr = attrs->GetInt(3);
    layout = layoutPtr != nullptr ? *layoutPtr : 0;

    OP_CHECK_IF(l < 1, OP_LOGE(context, "AreaFlashAttention: l must >= 1, got %ld", l), return ge::GRAPH_FAILED);
    OP_CHECK_IF(l > seqLength, OP_LOGE(context, "AreaFlashAttention: l=%ld > seqLength=%ld", l, seqLength), return ge::GRAPH_FAILED);
    OP_CHECK_IF(seqLength % l != 0,
        OP_LOGE(context, "AreaFlashAttention: seqLength %ld must be divisible by l %ld", seqLength, l),
        return ge::GRAPH_FAILED);

    if (direction != DIRECTION_H && direction != DIRECTION_W) {
        OP_LOGE(context, "AreaFlashAttention: invalid direction %ld, must be 0 (h) or 1 (w)", direction);
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(scale <= 0, OP_LOGE(context, "AreaFlashAttention: scale must > 0, got %f", scale), return ge::GRAPH_FAILED);

    OP_CHECK_IF(headDim % ALIGNMENT != 0,
        OP_LOGE(context, "AreaFlashAttention: headDim %ld should be multiple of %ld", headDim, ALIGNMENT),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void CalculateOptimalBlockSize(
    uint64_t ubSize,
    int64_t headDim,
    size_t typeSize,
    int64_t& blockBr,
    int64_t& blockBc)
{
    int64_t maxBr = 128;
    int64_t maxBc = 128;
    int64_t minBr = 16;
    int64_t minBc = 16;

    uint64_t availableUb = static_cast<uint64_t>(ubSize * UB_USAGE_RATIO);

    for (int64_t br = maxBr; br >= minBr; br -= 16) {
        for (int64_t bc = maxBc; bc >= minBc; bc -= 16) {
            uint64_t ubRequired =
                br * headDim * typeSize +
                bc * headDim * typeSize +
                bc * headDim * typeSize +
                br * headDim * typeSize +
                br * bc * sizeof(float) +
                br * sizeof(float) +
                br * sizeof(float) +
                br * headDim * sizeof(float);

            if (ubRequired <= availableUb) {
                blockBr = br;
                blockBc = bc;
                return;
            }
        }
    }

    blockBr = 16;
    blockBc = 16;
}

ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AreaFlashAttentionTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    int64_t batchSize, numHeads, seqLength, headDim, l, direction, layout;
    double scale;
    ge::DataType dataType;

    OP_CHECK_IF(
        GetShapeAttrsInfo(context, batchSize, numHeads, seqLength, headDim, l, direction, scale, layout, dataType) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    AreaFlashAttentionTilingData* tiling = context->GetTilingData<AreaFlashAttentionTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(AreaFlashAttentionTilingData), 0, sizeof(AreaFlashAttentionTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"),
        return ge::GRAPH_FAILED);

    int64_t seqLengthPerRegion = seqLength / l;

    size_t typeSize = 2;
    if (dataType == ge::DT_FLOAT) {
        typeSize = 4;
    } else if (dataType == ge::DT_BF16) {
        typeSize = 2;
    }

    int64_t blockBr, blockBc;
    CalculateOptimalBlockSize(ubSize, headDim, typeSize, blockBr, blockBc);

    int64_t totalRegions = batchSize * numHeads * l;
    int64_t blockDim = std::min(totalRegions, coreNum);

    tiling->batchSize = batchSize;
    tiling->numHeads = numHeads;
    tiling->seqLength = seqLength;
    tiling->headDim = headDim;
    tiling->regionNum = l;
    tiling->direction = direction;
    tiling->scale = scale;
    tiling->layout = layout;
    tiling->blockBr = blockBr;
    tiling->blockBc = blockBc;
    tiling->seqLengthPerRegion = seqLengthPerRegion;
    tiling->totalElements = batchSize * numHeads * seqLength * headDim;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    } else if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
    } else if (dataType == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_2);
    } else {
        OP_LOGE(context, "get dtype error");
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(tilingKey);

    OP_LOGD(context->GetNodeName(),
        "AreaFlashAttention Tiling: B=%ld, N=%ld, S=%ld, D=%ld, l=%ld, direction=%ld, "
        "Br=%ld, Bc=%ld, S_region=%ld, blockDim=%ld",
        batchSize, numHeads, seqLength, headDim, l, direction,
        blockBr, blockBc, seqLengthPerRegion, blockDim);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAreaFlashAttention([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AreaFlashAttention)
    .Tiling(AreaFlashAttentionTilingFunc)
    .TilingParse<AreaFlashAttentionCompileInfo>(TilingParseForAreaFlashAttention);

} // namespace optiling
