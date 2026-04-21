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
 * \file area_flash_attention.h
 * \brief AreaFlashAttention Kernel 实现
 */
#ifndef AREA_FLASH_ATTENTION_H
#define AREA_FLASH_ATTENTION_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "area_flash_attention_tiling_data.h"
#include "area_flash_attention_tiling_key.h"

namespace NsAreaFlashAttention {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TMP_BUFFER_NUM = 4;
constexpr float NEG_INF = -1e38f;

template <typename T>
class AreaFlashAttention
{
public:
    __aicore__ inline AreaFlashAttention(){};

    __aicore__ inline void Init(
        GM_ADDR query,
        GM_ADDR key,
        GM_ADDR value,
        GM_ADDR attentionOut,
        const AreaFlashAttentionTilingData* tilingData);
    
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessRegion(int64_t batchIdx, int64_t headIdx, int64_t regionIdx);
    __aicore__ inline void FlashAttentionBlock(
        LocalTensor<T>& qBlock,
        LocalTensor<T>& oBlock,
        LocalTensor<float>& maxLocal,
        LocalTensor<float>& sumLocal,
        int64_t qStartIdx);
    
    __aicore__ inline void CopyInQBlock(int64_t qStartIdx, LocalTensor<T>& qBlock);
    __aicore__ inline void CopyInKVBlock(int64_t kvStartIdx, LocalTensor<T>& kBlock, LocalTensor<T>& vBlock);
    __aicore__ inline void CopyOutOBlock(int64_t oStartIdx, LocalTensor<T>& oBlock);

private:
    TPipe pipe;
    
    TQue<QuePosition::VECIN, BUFFER_NUM> qQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> kQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> vQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> oQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> attnQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> maxQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> sumQueue;
    TQue<QuePosition::VECIN, TMP_BUFFER_NUM> tempQueue;

    GlobalTensor<T> queryGM;
    GlobalTensor<T> keyGM;
    GlobalTensor<T> valueGM;
    GlobalTensor<T> outputGM;

    int64_t batchSize_ = 0;
    int64_t numHeads_ = 0;
    int64_t seqLength_ = 0;
    int64_t headDim_ = 0;
    int64_t regionNum_ = 0;
    int64_t direction_ = 0;
    double scale_ = 1.0;
    int64_t layout_ = 0;
    int64_t blockBr_ = 0;
    int64_t blockBc_ = 0;
    int64_t seqLengthPerRegion_ = 0;
    int64_t totalElements_ = 0;
};

template <typename T>
__aicore__ inline void AreaFlashAttention<T>::Init(
    GM_ADDR query,
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR attentionOut,
    const AreaFlashAttentionTilingData* tilingData)
{
    batchSize_ = tilingData->batchSize;
    numHeads_ = tilingData->numHeads;
    seqLength_ = tilingData->seqLength;
    headDim_ = tilingData->headDim;
    regionNum_ = tilingData->regionNum;
    direction_ = tilingData->direction;
    scale_ = tilingData->scale;
    layout_ = tilingData->layout;
    blockBr_ = tilingData->blockBr;
    blockBc_ = tilingData->blockBc;
    seqLengthPerRegion_ = tilingData->seqLengthPerRegion;
    totalElements_ = tilingData->totalElements;

    queryGM.SetGlobalBuffer((__gm__ T*)query, totalElements_);
    keyGM.SetGlobalBuffer((__gm__ T*)key, totalElements_);
    valueGM.SetGlobalBuffer((__gm__ T*)value, totalElements_);
    outputGM.SetGlobalBuffer((__gm__ T*)attentionOut, totalElements_);

    pipe.InitBuffer(qQueue, BUFFER_NUM, blockBr_ * headDim_ * sizeof(T));
    pipe.InitBuffer(kQueue, BUFFER_NUM, blockBc_ * headDim_ * sizeof(T));
    pipe.InitBuffer(vQueue, BUFFER_NUM, blockBc_ * headDim_ * sizeof(T));
    pipe.InitBuffer(oQueue, BUFFER_NUM, blockBr_ * headDim_ * sizeof(T));
    pipe.InitBuffer(attnQueue, BUFFER_NUM, blockBr_ * blockBc_ * sizeof(float));
    pipe.InitBuffer(maxQueue, BUFFER_NUM, blockBr_ * sizeof(float));
    pipe.InitBuffer(sumQueue, BUFFER_NUM, blockBr_ * sizeof(float));
    pipe.InitBuffer(tempQueue, TMP_BUFFER_NUM, blockBr_ * headDim_ * sizeof(float));
}

template <typename T>
__aicore__ inline void AreaFlashAttention<T>::CopyInQBlock(int64_t qStartIdx, LocalTensor<T>& qBlock)
{
    DataCopy(qBlock, queryGM[qStartIdx * headDim_], blockBr_ * headDim_);
}

template <typename T>
__aicore__ inline void AreaFlashAttention<T>::CopyInKVBlock(
    int64_t kvStartIdx,
    LocalTensor<T>& kBlock,
    LocalTensor<T>& vBlock)
{
    DataCopy(kBlock, keyGM[kvStartIdx * headDim_], blockBc_ * headDim_);
    DataCopy(vBlock, valueGM[kvStartIdx * headDim_], blockBc_ * headDim_);
}

template <typename T>
__aicore__ inline void AreaFlashAttention<T>::CopyOutOBlock(int64_t oStartIdx, LocalTensor<T>& oBlock)
{
    DataCopy(outputGM[oStartIdx * headDim_], oBlock, blockBr_ * headDim_);
}

template <typename T>
__aicore__ inline void AreaFlashAttention<T>::FlashAttentionBlock(
    LocalTensor<T>& qBlock,
    LocalTensor<T>& oBlock,
    LocalTensor<float>& maxLocal,
    LocalTensor<float>& sumLocal,
    int64_t qStartIdx)
{
    Duplicate(maxLocal, NEG_INF, blockBr_);
    Duplicate(sumLocal, 0.0f, blockBr_);
    Duplicate(oBlock, (T)0, blockBr_ * headDim_);

    for (int64_t j = 0; j < seqLengthPerRegion_; j += blockBc_) {
        int64_t actualBc = min(blockBc_, seqLengthPerRegion_ - j);
        
        LocalTensor<T> kBlock = kQueue.AllocTensor<T>();
        LocalTensor<T> vBlock = vQueue.AllocTensor<T>();
        CopyInKVBlock(j, kBlock, vBlock);
        
        LocalTensor<float> attnScores = attnQueue.AllocTensor<float>();
        
        LocalTensor<float> qFloat = tempQueue.AllocTensor<float>();
        LocalTensor<float> kFloat = tempQueue.AllocTensor<float>();
        Cast(qFloat, qBlock, RoundMode::CAST_NONE, blockBr_ * headDim_);
        Cast(kFloat, kBlock, RoundMode::CAST_NONE, actualBc * headDim_);
        
        for (int64_t i = 0; i < blockBr_; i++) {
            for (int64_t kj = 0; kj < actualBc; kj++) {
                float score = 0.0f;
                for (int64_t d = 0; d < headDim_; d++) {
                    score += qFloat.GetValue(i * headDim_ + d) * kFloat.GetValue(kj * headDim_ + d);
                }
                score *= static_cast<float>(scale_);
                attnScores.SetValue(i * actualBc + kj, score);
            }
        }
        
        LocalTensor<float> maxNew = tempQueue.AllocTensor<float>();
        LocalTensor<float> sumNew = tempQueue.AllocTensor<float>();
        
        for (int64_t i = 0; i < blockBr_; i++) {
            float rowMax = NEG_INF;
            for (int64_t kj = 0; kj < actualBc; kj++) {
                float score = attnScores.GetValue(i * actualBc + kj);
                if (score > rowMax) rowMax = score;
            }
            float oldMax = maxLocal.GetValue(i);
            float newMax = max(oldMax, rowMax);
            maxNew.SetValue(i, newMax);
            
            float scaleOld = exp(oldMax - newMax);
            float sumExp = 0.0f;
            for (int64_t kj = 0; kj < actualBc; kj++) {
                float score = attnScores.GetValue(i * actualBc + kj);
                float prob = exp(score - newMax);
                attnScores.SetValue(i * actualBc + kj, prob);
                sumExp += prob;
            }
            float oldSum = sumLocal.GetValue(i);
            sumNew.SetValue(i, scaleOld * oldSum + sumExp);
        }
        
        LocalTensor<float> vFloat = tempQueue.AllocTensor<float>();
        LocalTensor<float> oFloat = tempQueue.AllocTensor<float>();
        Cast(vFloat, vBlock, RoundMode::CAST_NONE, actualBc * headDim_);
        Cast(oFloat, oBlock, RoundMode::CAST_NONE, blockBr_ * headDim_);
        
        for (int64_t i = 0; i < blockBr_; i++) {
            float oldMax = maxLocal.GetValue(i);
            float newMax = maxNew.GetValue(i);
            float scaleOld = exp(oldMax - newMax);
            
            for (int64_t d = 0; d < headDim_; d++) {
                float oVal = oFloat.GetValue(i * headDim_ + d) * scaleOld;
                for (int64_t kj = 0; kj < actualBc; kj++) {
                    float prob = attnScores.GetValue(i * actualBc + kj);
                    oVal += prob * vFloat.GetValue(kj * headDim_ + d);
                }
                oFloat.SetValue(i * headDim_ + d, oVal);
            }
        }
        
        Cast(oBlock, oFloat, RoundMode::CAST_RINT, blockBr_ * headDim_);
        
        for (int64_t i = 0; i < blockBr_; i++) {
            maxLocal.SetValue(i, maxNew.GetValue(i));
            sumLocal.SetValue(i, sumNew.GetValue(i));
        }
        
        kQueue.FreeTensor(kBlock);
        vQueue.FreeTensor(vBlock);
        attnQueue.FreeTensor(attnScores);
        tempQueue.FreeTensor(qFloat);
        tempQueue.FreeTensor(kFloat);
        tempQueue.FreeTensor(maxNew);
        tempQueue.FreeTensor(sumNew);
        tempQueue.FreeTensor(vFloat);
        tempQueue.FreeTensor(oFloat);
    }
    
    LocalTensor<float> invSum = tempQueue.AllocTensor<float>();
    for (int64_t i = 0; i < blockBr_; i++) {
        float sum = sumLocal.GetValue(i);
        invSum.SetValue(i, 1.0f / sum);
    }
    
    LocalTensor<float> oFloat = tempQueue.AllocTensor<float>();
    Cast(oFloat, oBlock, RoundMode::CAST_NONE, blockBr_ * headDim_);
    
    for (int64_t i = 0; i < blockBr_; i++) {
        float inv = invSum.GetValue(i);
        for (int64_t d = 0; d < headDim_; d++) {
            float val = oFloat.GetValue(i * headDim_ + d) * inv;
            oFloat.SetValue(i * headDim_ + d, val);
        }
    }
    
    Cast(oBlock, oFloat, RoundMode::CAST_RINT, blockBr_ * headDim_);
    tempQueue.FreeTensor(invSum);
    tempQueue.FreeTensor(oFloat);
}

template <typename T>
__aicore__ inline void AreaFlashAttention<T>::ProcessRegion(
    int64_t batchIdx,
    int64_t headIdx,
    int64_t regionIdx)
{
    int64_t regionOffset = ((batchIdx * numHeads_ + headIdx) * seqLength_ + 
                            regionIdx * seqLengthPerRegion_) * headDim_;
    
    for (int64_t i = 0; i < seqLengthPerRegion_; i += blockBr_) {
        int64_t actualBr = min(blockBr_, seqLengthPerRegion_ - i);
        
        LocalTensor<T> qBlock = qQueue.AllocTensor<T>();
        LocalTensor<T> oBlock = oQueue.AllocTensor<T>();
        LocalTensor<float> maxLocal = maxQueue.AllocTensor<float>();
        LocalTensor<float> sumLocal = sumQueue.AllocTensor<float>();
        
        CopyInQBlock(regionOffset / headDim_ + i, qBlock);
        
        FlashAttentionBlock(qBlock, oBlock, maxLocal, sumLocal, i);
        
        CopyOutOBlock(regionOffset / headDim_ + i, oBlock);
        
        qQueue.FreeTensor(qBlock);
        oQueue.FreeTensor(oBlock);
        maxQueue.FreeTensor(maxLocal);
        sumQueue.FreeTensor(sumLocal);
    }
}

template <typename T>
__aicore__ inline void AreaFlashAttention<T>::Process()
{
    int64_t blockIdx = GetBlockIdx();
    int64_t totalRegions = batchSize_ * numHeads_ * regionNum_;
    int64_t blockDim = GetBlockNum();
    
    for (int64_t regionIdx = blockIdx; regionIdx < totalRegions; regionIdx += blockDim) {
        int64_t batchIdx = regionIdx / (numHeads_ * regionNum_);
        int64_t remaining = regionIdx % (numHeads_ * regionNum_);
        int64_t headIdx = remaining / regionNum_;
        int64_t region = remaining % regionNum_;
        
        ProcessRegion(batchIdx, headIdx, region);
    }
}

} // namespace NsAreaFlashAttention

#endif // AREA_FLASH_ATTENTION_H
