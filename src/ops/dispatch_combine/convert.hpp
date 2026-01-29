// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <cstdint>

#include <hip/hip_fp16.h>

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"

namespace mori {
namespace moe {

struct ConvertDispatchOutputArgs {
  EpDispatchCombineConfig config;
  const void* dispatchOutX{nullptr};
  const void* dispatchOutTopkIdx{nullptr};
  const index_t* dispatchSrcTokenPos{nullptr};
  const index_t* totalRecvTokenNum{nullptr};
  uint32_t* dispatchGridBarrier{nullptr};
  void* packedRecvX{nullptr};
  void* packedRecvCount{nullptr};
  void* packedRecvSrcInfo{nullptr};
  void* packedRecvLayoutRange{nullptr};
  uint64_t* dispTokToEpSlotMap{nullptr};
};

struct ConvertCombineInputArgs {
  EpDispatchCombineConfig config;
  const void* packedRecvX{nullptr};
  const void* topkIdx{nullptr};
  const void* topkWeights{nullptr};
  const void* packedRecvSrcInfo{nullptr};
  const void* packedRecvLayoutRange{nullptr};
  const index_t* totalRecvTokenNum{nullptr};
  void* combineInput{nullptr};
  uint64_t* dispTokToEpSlotMap{nullptr};
};

__device__ inline void ConvertDispatchOutputDevice(ConvertDispatchOutputArgs args) {
  const EpDispatchCombineConfig& config = args.config;
  const int thdId = threadIdx.x;
  const int warpId = thdId / warpSize;
  const int laneId = thdId & (warpSize - 1);
  const int warpNum = blockDim.x / warpSize;

  const int globalWarpId = blockIdx.x * warpNum + warpId;
  const int globalWarpNum = gridDim.x * warpNum;

  const int topk = config.numExpertPerToken;

  const int64_t maxNumTokenPerRank = config.maxNumInpTokenPerRank;
  const int64_t maxTokensPerExpert =
      static_cast<int64_t>(config.worldSize) * config.maxNumInpTokenPerRank;
  const size_t hiddenBytes = static_cast<size_t>(config.hiddenDim) * config.maxTokenTypeSize;

  const auto* topkIdx = reinterpret_cast<const index_t*>(args.dispatchOutTopkIdx);
  const auto* dispatchSrcTokenPos = args.dispatchSrcTokenPos;
  auto* packedRecvX = reinterpret_cast<uint8_t*>(args.packedRecvX);
  auto* packedRecvSrcInfo = reinterpret_cast<index_t*>(args.packedRecvSrcInfo);
  auto* packedRecvCount = reinterpret_cast<uint32_t*>(args.packedRecvCount);
  (void)args.packedRecvLayoutRange;
  auto* dispTokToEpSlotMap = args.dispTokToEpSlotMap;

  if (thdId < args.config.numExpertPerRank) {
    packedRecvCount[thdId] = 0;
  }
  uint32_t* dispatchGridBarrier = args.dispatchGridBarrier + 1;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    __hip_atomic_fetch_add(dispatchGridBarrier, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    while (atomicCAS(dispatchGridBarrier, gridDim.x, 0) != 0) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();

  const int64_t totalTokens = static_cast<int64_t>(args.totalRecvTokenNum[0]);
  for (int i = globalWarpId; i < totalTokens * topk; i += globalWarpNum) {
    auto tokenIdx = i / topk;
    const index_t expertId = topkIdx[i];
    const auto localExpert = expertId - config.rank * config.numExpertPerRank;
    if (localExpert < 0 || localExpert >= config.numExpertPerRank) {
      if (laneId == 0) {
        // TODO: do not use -1
        dispTokToEpSlotMap[i] = static_cast<uint64_t>(-1);
      }
      continue;
    }

    const index_t srcTokenPos = dispatchSrcTokenPos[tokenIdx];
    const int srcRank = static_cast<int>(srcTokenPos / maxNumTokenPerRank);
    const int srcInfo = static_cast<int>(srcTokenPos - srcRank * maxNumTokenPerRank);
    if (srcRank < 0 || srcRank >= config.worldSize) {
      continue;
    }

    uint32_t idx = 0;
    if (laneId == 0) {
      idx = atomicAdd(packedRecvCount + localExpert, 1u);
#if 0
      if (config.rank == 0 || config.rank == 1) {
        printf("[ConvertDispatchOutputKernel] rank=%d tokenIdx=%lld localExpert=%lld idx=%u\n",
               config.rank, static_cast<long long>(tokenIdx), static_cast<long long>(localExpert), idx);
      }
#endif
    }
    idx = __shfl(idx, 0);

    const uint64_t linearIndex =
        static_cast<uint64_t>(localExpert) * maxTokensPerExpert + idx;
    if (laneId == 0) {
      // packedRecvSrcInfo[linearIndex] = srcInfo;
      packedRecvSrcInfo[linearIndex] = srcTokenPos;
      dispTokToEpSlotMap[i] = linearIndex;
    }

    const size_t dstOffset = static_cast<size_t>(linearIndex) * hiddenBytes;
    const size_t srcOffset = static_cast<size_t>(tokenIdx) * hiddenBytes;
    const auto* srcBytes = reinterpret_cast<const uint8_t*>(args.dispatchOutX) + srcOffset;
    auto* dstBytes = packedRecvX + dstOffset;
    core::WarpCopy<uint8_t>(dstBytes, srcBytes, hiddenBytes);
  }
}

__global__ void ConvertDispatchOutputKernel(ConvertDispatchOutputArgs args) {
  ConvertDispatchOutputDevice(args);
}

template <typename T>
__global__ void ConvertCombineInputKernel(ConvertCombineInputArgs args) {
  const EpDispatchCombineConfig& config = args.config;
  const int thdId = threadIdx.x;
  const int warpId = thdId / warpSize;
  const int warpNum = blockDim.x / warpSize;

  const int globalWarpId = blockIdx.x * warpNum + warpId;
  const int globalWarpNum = gridDim.x * warpNum;

  const int topk = config.numExpertPerToken;
  const int64_t hiddenDim = config.hiddenDim;

  auto* dispTokToEpSlotMap = args.dispTokToEpSlotMap;
  const auto* packedRecvX = reinterpret_cast<const T*>(args.packedRecvX);
  auto* combineInput = reinterpret_cast<T*>(args.combineInput);
  const auto* topkWeights = reinterpret_cast<const float*>(args.topkWeights);

  T* srcPtrs[MAX_EXPERTS_PER_TOKEN];

  const int64_t totalTokens = static_cast<int64_t>(args.totalRecvTokenNum[0]);
  for (int64_t tokenIdx = globalWarpId; tokenIdx < totalTokens; tokenIdx += globalWarpNum) {
    for (int k = 0; k < topk; ++k) {
      const uint64_t slot = dispTokToEpSlotMap[tokenIdx * topk + k];
      srcPtrs[k] = (slot == static_cast<uint64_t>(-1))
                       ? nullptr
                       : const_cast<T*>(packedRecvX + slot * hiddenDim);
    }

    const float* weightRow = topkWeights ? (topkWeights + tokenIdx * topk) : nullptr;
    core::WarpAccum<T, 4>(combineInput + tokenIdx * hiddenDim, srcPtrs, weightRow, topk,
                          hiddenDim);
  }
}

}  // namespace moe
}  // namespace mori
