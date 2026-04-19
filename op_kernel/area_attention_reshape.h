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
 * \file area_attention_reshape.h
 * \brief Area Attention Reshape Kernel Implementation
 */
#ifndef AREA_ATTENTION_RESHAPE_H
#define AREA_ATTENTION_RESHAPE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "area_attention_reshape_tiling_data.h"
#include "area_attention_reshape_tiling_key.h"

namespace NsAreaAttentionReshape {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t DIRECTION_H = 0;
constexpr int64_t DIRECTION_W = 1;

template <typename T>
class AreaAttentionReshape
{
public:
    __aicore__ inline AreaAttentionReshape(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AreaAttentionReshapeTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    int64_t batchSize_ = 0;
    int64_t height_ = 0;
    int64_t width_ = 0;
    int64_t channels_ = 0;
    int64_t l_ = 0;
    int64_t direction_ = 0;
    int64_t totalElements_ = 0;
    int64_t tileNum_ = 0;
    int64_t elementsPerTile_ = 0;
    int64_t elementsPerBlock_ = 0;
};

template <typename T>
__aicore__ inline void AreaAttentionReshape<T>::Init(GM_ADDR x, GM_ADDR y, const AreaAttentionReshapeTilingData* tilingData)
{
    batchSize_ = tilingData->batchSize;
    height_ = tilingData->height;
    width_ = tilingData->width;
    channels_ = tilingData->channels;
    l_ = tilingData->l;
    direction_ = tilingData->direction;
    totalElements_ = tilingData->totalElements;
    tileNum_ = tilingData->tileNum;

    elementsPerBlock_ = totalElements_ / AscendC::GetBlockNum();
    elementsPerTile_ = elementsPerBlock_ / tileNum_;

    inputGM.SetGlobalBuffer((__gm__ T*)x, totalElements_);
    outputGM.SetGlobalBuffer((__gm__ T*)y, totalElements_);

    pipe.InitBuffer(inputQueue, BUFFER_NUM, elementsPerTile_ * sizeof(T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, elementsPerTile_ * sizeof(T));
}

template <typename T>
__aicore__ inline void AreaAttentionReshape<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> inputLocal = inputQueue.AllocTensor<T>();
    int64_t offset = AscendC::GetBlockIdx() * elementsPerBlock_ + progress * elementsPerTile_;
    AscendC::DataCopy(inputLocal, inputGM[offset], elementsPerTile_);
    inputQueue.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void AreaAttentionReshape<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> outputLocal = outputQueue.DeQue<T>();
    int64_t offset = AscendC::GetBlockIdx() * elementsPerBlock_ + progress * elementsPerTile_;
    AscendC::DataCopy(outputGM[offset], outputLocal, elementsPerTile_);
    outputQueue.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void AreaAttentionReshape<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> inputLocal = inputQueue.DeQue<T>();
    AscendC::LocalTensor<T> outputLocal = outputQueue.AllocTensor<T>();

    int64_t tileStartIdx = AscendC::GetBlockIdx() * elementsPerBlock_ + progress * elementsPerTile_;
    
    for (int64_t i = 0; i < elementsPerTile_; i++) {
        int64_t globalIdx = tileStartIdx + i;
        
        int64_t c = globalIdx % channels_;
        int64_t rest = globalIdx / channels_;
        int64_t w = rest % width_;
        rest = rest / width_;
        int64_t h = rest % height_;
        int64_t b = rest / height_;

        int64_t outIdx;
        if (direction_ == DIRECTION_H) {
            int64_t h_chunk = height_ / l_;
            int64_t strip_id = h / h_chunk;
            int64_t h_in_strip = h % h_chunk;
            int64_t new_b = b * l_ + strip_id;
            int64_t n_chunk = w * h_chunk + h_in_strip;
            outIdx = (new_b * (width_ * h_chunk) + n_chunk) * channels_ + c;
        } else {
            int64_t w_chunk = width_ / l_;
            int64_t strip_id = w / w_chunk;
            int64_t w_in_strip = w % w_chunk;
            int64_t new_b = b * l_ + strip_id;
            int64_t n_chunk = h * w_chunk + w_in_strip;
            outIdx = (new_b * (height_ * w_chunk) + n_chunk) * channels_ + c;
        }

        int64_t outTileIdx = outIdx / elementsPerTile_;
        int64_t outOffsetInTile = outIdx % elementsPerTile_;
        
        if (outTileIdx == progress) {
            outputLocal.SetValue(outOffsetInTile, inputLocal.GetValue(i));
        }
    }

    outputQueue.EnQue<T>(outputLocal);
    inputQueue.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void AreaAttentionReshape<T>::Process()
{
    for (int32_t i = 0; i < tileNum_; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsAreaAttentionReshape
#endif // AREA_ATTENTION_RESHAPE_H
