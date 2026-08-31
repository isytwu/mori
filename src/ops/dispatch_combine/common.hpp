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

#include "mori/ops/dispatch_combine/dispatch_combine.hpp"

namespace mori {
namespace moe {

// Encode/decode a (pe, localTokId) pair into a flat global token index for bookkeeping.
// Stride: MaxNumTokensToSend(), which guarantees uniqueness across all PEs.
inline __device__ int FlatTokenIndex(const EpDispatchCombineConfig& config, int pe,
                                     int localTokId) {
  return pe * config.MaxNumTokensToSend() + localTokId;
}
inline __device__ int PeFromFlatTokenIndex(const EpDispatchCombineConfig& config, int flatIdx) {
  return flatIdx / config.MaxNumTokensToSend();
}
inline __device__ int LocalTokIdFromFlatTokenIndex(const EpDispatchCombineConfig& config,
                                                   int flatIdx) {
  return flatIdx % config.MaxNumTokensToSend();
}
inline __device__ int NullFlatTokenIndex(const EpDispatchCombineConfig& config) {
  return config.worldSize * config.MaxNumTokensToSend();
}

// Encode/decode a flat offset into a per-PE send staging buffer.
// Stride: MaxNumTokensToSendPerRank, which determines how much buffer space is allocated per PE.
// NullSendBufSlotOffset() returns an out-of-range sentinel indicating "already sent".
inline __device__ int SendBufSlotOffset(const EpDispatchCombineConfig& config, int pe, int slotId) {
  return pe * config.MaxNumTokensToSendPerRank() + slotId;
}
inline __device__ int PeFromSendBufSlotOffset(const EpDispatchCombineConfig& config, int flatIdx) {
  return flatIdx / config.MaxNumTokensToSendPerRank();
}
inline __device__ int SlotIdFromSendBufSlotOffset(const EpDispatchCombineConfig& config,
                                                  int flatIdx) {
  return flatIdx % config.MaxNumTokensToSendPerRank();
}
inline __device__ int NullSendBufSlotOffset(const EpDispatchCombineConfig& config) {
  return config.worldSize * config.MaxNumTokensToSendPerRank();
}

// ---------------------------------------------------------------------------
// P2P-write ("push") combine staging.
//
// In the default pull combine, a PE gathering one of its tokens reads each
// expert's partial straight out of that expert PE's combineInp. The read is an
// xGMI round trip per (token, expert) and its *latency* is what caps the small-
// token combine. The push variant flips it: in CombineSync each expert PE
// writes its partial into the *reader's* combineInp, so the gather is all local
// loads.
//
// A pushed slot has to be addressable by both sides without any extra
// bookkeeping (dispatch is untouched, so the only thing the writer has is
// dispTokIdToSrcTokId -> the token's original owner). The key is:
//
//   (writerPe, ownerNode, ownerLocalTokId)
//
// which is unique on any given reader PE R:
//   - intra-node origin: R *is* the owner, ownerNode == R's node, and
//     ownerLocalTokId is R's own token id. Dispatch dedups per destination PE,
//     so at most one push lands per (writerPe, tokenId).
//   - inter-node origin: R is the node aggregator, i.e. the PE that received
//     the token over RDMA. Dispatch always sends to
//     `destNode * gpuPerNode + (ownerPe % gpuPerNode)` (see the proxyPe in
//     internode_v1.cpp), so for a fixed R and a fixed ownerNode the owner PE is
//     uniquely determined, and (ownerNode, ownerLocalTokId) names the token.
//   - the two never alias: intra origin has ownerNode == myNode, inter origin
//     has ownerNode != myNode.
//
// Stride is MaxNumTokensToSendPerRank (the owner's token space), NOT
// MaxNumTokensToRecvPerRank -- the latter can be smaller when maxTotalRecvTokens
// is set, which would make distinct owner tokens alias.
// Slot count lives on the config (EpDispatchCombineConfig::CombinePushSlotNum) so the
// host-only allocator in dispatch_combine.cpp can size the buffers with it.
inline __device__ int CombinePushSlot(const EpDispatchCombineConfig& config, int writerPe,
                                      int ownerNode, int ownerLocalTokId) {
  const int nNodes = config.worldSize / config.gpuPerNode;
  // The JIT build does not pass -DNDEBUG, so these stay live and catch a
  // mis-sized combineInp (the failure mode is a silent OOB write into a peer's
  // heap, which shows up much later as unrelated corruption).
  assert((writerPe >= 0) && (writerPe < config.worldSize));
  assert((ownerNode >= 0) && (ownerNode < nNodes));
  assert((ownerLocalTokId >= 0) && (ownerLocalTokId < config.MaxNumTokensToSendPerRank()));
  int slot = config.CombinePushSlotBase() +
             (writerPe * nNodes + ownerNode) * config.MaxNumTokensToSendPerRank() + ownerLocalTokId;
  assert(slot < config.CombineStagingSlotNum());
  return slot;
}

// Partitions a loop over (numItems x dimSize) work across globalWarpNum warps.
// When there are more warps than items, multiple warps collaborate on a single item
// by splitting dimSize; when there are fewer warps, each warp handles multiple items.
struct MultiWarpIter {
  int warpsPerItem;
  size_t dimPerWarp;
  size_t dimSize;

  // dimGranularity rounds dimPerWarp up to a multiple of itself, so callers doing
  // vectorized loads get every warp's slice starting on a vector boundary and
  // sized in whole vector steps.
  inline __device__ MultiWarpIter(int globalWarpNum, int numItems, size_t dimSize_,
                                  size_t dimGranularity = 1)
      : dimSize(dimSize_) {
    warpsPerItem = (globalWarpNum + numItems - 1) / numItems;
    dimPerWarp = (dimSize + warpsPerItem - 1) / warpsPerItem;
    if (dimGranularity > 1) {
      dimPerWarp = ((dimPerWarp + dimGranularity - 1) / dimGranularity) * dimGranularity;
      // A coarser slice means fewer warps are actually needed; keep warpsPerItem
      // consistent with dimPerWarp or the tail warps decode to empty ranges.
      warpsPerItem = static_cast<int>((dimSize + dimPerWarp - 1) / dimPerWarp);
    }
  }

  inline __device__ void Decode(int i, int& itemId, int& inItemPartId, size_t& dimOffset,
                                size_t& dimChunk) const {
    itemId = i / warpsPerItem;
    inItemPartId = i % warpsPerItem;
    dimOffset = (size_t)inItemPartId * dimPerWarp;
    dimChunk = (dimOffset < dimSize) ? std::min(dimSize - dimOffset, dimPerWarp) : size_t{0};
  }
};

#define DEF_COMMON_VARS                                    \
  const EpDispatchCombineConfig& config = args.config;     \
  int thdId = threadIdx.x;                                 \
  int thdNum = blockDim.x;                                 \
  int laneId = threadIdx.x & (warpSize - 1);               \
  int warpId = thdId / warpSize;                           \
  int warpNum = blockDim.x / warpSize;                     \
  int blockNum = gridDim.x;                                \
  int blockId = blockIdx.x;                                \
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x; \
  int globalThdNum = gridDim.x * blockDim.x;               \
  int globalWarpId = blockIdx.x * warpNum + warpId;        \
  int globalWarpNum = gridDim.x * warpNum;                 \
  int nullTokenId = NullFlatTokenIndex(config);            \
  int myPe = config.rank;                                  \
  int npes = config.worldSize;                             \
  int myNode = myPe / config.gpuPerNode;                   \
  int nNodes = npes / config.gpuPerNode;                   \
  int numExpertPerToken = config.numExpertPerToken;        \
  assert(numExpertPerToken < warpSize);                    \
  size_t hiddenDim = config.HiddenDimSz();                 \
  size_t hiddenBytes = config.HiddenBytes(sizeof(T));      \
  size_t indexBytes = config.IndexBytes();                 \
  size_t weightBytes = config.WeightBytes();               \
  size_t srcTokenIdBytes = config.SrcTokenIdBytes();       \
  size_t scaleBytes = config.ScaleBytes();                 \
  size_t xferBytes = config.XferBytesPerToken(sizeof(T));  \
  size_t combXferBytes = (args.weightsBuf == nullptr) ? hiddenBytes : hiddenBytes + weightBytes;

}  // namespace moe
}  // namespace mori
