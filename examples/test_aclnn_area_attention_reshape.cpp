/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_area_attention_reshape.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < std::min(size, (int64_t)20); i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    LOG_PRINT("... (showing first 20 elements of %ld total)\n", size);
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    LOG_PRINT("=== Area Attention Reshape Test ===\n");
    
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 测试参数
    int64_t B = 2;      // batch size
    int64_t H = 8;      // height
    int64_t W = 8;      // width
    int64_t C = 16;     // channels
    int64_t l = 4;      // 切分段数
    int64_t direction = 0;  // 0: 高度方向, 1: 宽度方向

    LOG_PRINT("Input shape: (%ld, %ld, %ld, %ld)\n", B, H, W, C);
    LOG_PRINT("l = %ld, direction = %s\n", l, direction == 0 ? "height" : "width");

    // 创建输入张量
    aclTensor* inputTensor = nullptr;
    void* inputDeviceAddr = nullptr;
    std::vector<int64_t> inputShape = {B, H, W, C};
    std::vector<float> inputHostData(B * H * W * C, 1.0f);
    
    // 填充一些测试数据
    for (int64_t i = 0; i < B * H * W * C; i++) {
        inputHostData[i] = static_cast<float>(i % 100);
    }
    
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &inputTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输出张量
    aclTensor* outputTensor = nullptr;
    void* outputDeviceAddr = nullptr;
    int64_t newB = B * l;
    int64_t nChunk = (direction == 0) ? (W * (H / l)) : (H * (W / l));
    std::vector<int64_t> outputShape = {newB, nChunk, C};
    std::vector<float> outputHostData(newB * nChunk * C, 0.0f);
    
    LOG_PRINT("Output shape: (%ld, %ld, %ld)\n", newB, nChunk, C);
    
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &outputTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用算子
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnAreaAttentionReshapeGetWorkspaceSize(inputTensor, l, direction, outputTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAreaAttentionReshapeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnAreaAttentionReshape(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAreaAttentionReshape failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("\n=== Output Results ===\n");
    PrintOutResult(outputShape, &outputDeviceAddr);

    // 清理资源
    aclDestroyTensor(inputTensor);
    aclDestroyTensor(outputTensor);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("\n=== Test Completed Successfully ===\n");
    return 0;
}
