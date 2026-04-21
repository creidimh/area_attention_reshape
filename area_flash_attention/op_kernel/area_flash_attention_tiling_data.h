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
 * \file area_flash_attention_tiling_data.h
 * \brief AreaFlashAttention Tiling Data 结构定义
 */

#ifndef __AREA_FLASH_ATTENTION_TILLING_DATA_H__
#define __AREA_FLASH_ATTENTION_TILLING_DATA_H__

struct AreaFlashAttentionTilingData {
    int64_t batchSize;
    int64_t numHeads;
    int64_t seqLength;
    int64_t headDim;
    int64_t regionNum;
    int64_t direction;
    double scale;
    int64_t layout;
    int64_t blockBr;
    int64_t blockBc;
    int64_t seqLengthPerRegion;
    int64_t totalElements;
};

#endif
