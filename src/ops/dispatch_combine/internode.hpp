#pragma once

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace moe {

#define MAX_GPUS_PER_NODE 8

#define DEBUG 0

__device__ void SyncIfDebugEnabled(const char* msg) {
#if DEBUG == 1
  __syncthreads();
  if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
    shmem::ShmemQuietThread();
    printf("%s\n", msg);
  }
  __syncthreads();
#endif
}
//TODO 优化：少用block性能相同；atomic换imm？
/* ---------------------------------------------------------------------------------------------- */
/*                                    EpDispatchInterNodeKernel                                   */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpDispatchInterNodeKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;

  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalThdNum = gridDim.x * blockDim.x;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  int myPe = config.rank;
  int npes = config.worldSize;
  int myNode = myPe / MAX_GPUS_PER_NODE;

  size_t MaxNumTokensToSendPerRank = config.MaxNumTokensToSendPerRank();
  size_t MaxNumTokensToRecvPerRank = config.MaxNumTokensToRecvPerRank();
  size_t MaxNumTokensToRecv = config.MaxNumTokensToRecv();

  int numExpertPerToken = config.numExpertPerToken;
  assert(numExpertPerToken < warpSize);

  size_t weightOffset = config.hiddenDim * sizeof(T);
  size_t indicesOffset = weightOffset + sizeof(float) * numExpertPerToken;
  size_t scalesOffset = indicesOffset + sizeof(index_t) * numExpertPerToken;
  size_t stagingOffset = scalesOffset + config.scaleTypeSize * config.scaleDim;

  extern __shared__ char sharedMem[];

  int subWarpNumPerWarp = warpSize / numExpertPerToken;//32/topk，每个EP对应一个subwarp
  int laneInSubWarp = laneId % numExpertPerToken;
  int subWarpId = laneId / numExpertPerToken;
  int globalSubWarpId = globalWarpId * subWarpNumPerWarp + subWarpId;
  int globalSubWarpNum = globalWarpNum * subWarpNumPerWarp;
  if (laneId < subWarpNumPerWarp * numExpertPerToken) {
    for (int tokenId = globalSubWarpId; tokenId < args.curRankNumToken;
         tokenId += globalSubWarpNum) {
      const int expertOffset = tokenId * numExpertPerToken + laneInSubWarp;
      index_t destExpert = args.tokenIndices[expertOffset];
      index_t destPe = destExpert / config.numExpertPerRank;

      unsigned long long subWarpMask = ((1ULL << numExpertPerToken) - 1ULL)
                                       << (subWarpId * numExpertPerToken);
      unsigned long long dupMask = __match_any_sync(subWarpMask, destPe);
      bool dup = false;
      if (laneInSubWarp) {
        unsigned long long lowerMask =
            dupMask & (((1ULL << laneInSubWarp) - 1ULL) << (subWarpId * numExpertPerToken));
        dup = (lowerMask != 0ULL);
      }
      if (dup) {
        args.dispSenderIdxMap[expertOffset] = MaxNumTokensToRecv;
        continue;
      } else {
        index_t destPeTokenIdx = 0, peSortedIdx = 0;
        destPeTokenIdx = atomicAdd(args.destPeTokenCounter + destPe, 1);//和intranode不同，这里不是所有pe对于destPE取号
        peSortedIdx = destPe * MaxNumTokensToRecvPerRank + destPeTokenIdx;//写入staging的位置
        args.dispSenderIdxMap[expertOffset] = peSortedIdx;//给combine也输出给用户：topk_idx和staging位置
        args.destPeTokenIdxMap[peSortedIdx] = tokenId;  //给dispatch：input的tokenId拷贝到shmemStagingTokMemObj的peSortedIdx，也就是记录了src token的信息；因为后面chunk是按照发送的总token划分的，需要知道每个staging的位置从哪个token拷贝
        __threadfence();
      }
    }
  }  // 优化了去重逻辑，增加了subwarp，另外用了__match_any_sync；已经计算好了发给每个PE多少token-destPeTokenCounter，以及位置

  if (laneId == 0) {
    int old_val = atomicAdd(args.dispatchGridBarrier, 1);
    if (old_val == globalWarpNum - 1) {
      __hip_atomic_store(args.dispatchGridBarrier, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }

  if (laneId == 0) {
    shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, 0);
  }//所有warp的同步，TODO 会有bug：99行很多warp都在置0，连续做同步，其他进去下一轮的warp加1会被置0

  // TODO: block num should be multiple of npes
  const int numsBlockPerDestPe = gridDim.x / npes;
  const int destPe = blockIdx.x / numsBlockPerDestPe;
  const int destNode = destPe / MAX_GPUS_PER_NODE;
  const int localBlockId = blockIdx.x - destPe * numsBlockPerDestPe;
  const int totalTokens = args.destPeTokenCounter[destPe];//要发给destPe的总token
  const int baseChunk = totalTokens / numsBlockPerDestPe;
  const int remainder = totalTokens % numsBlockPerDestPe;

  const int myChunkSize = baseChunk + (localBlockId < remainder);

  const int startIdx = localBlockId * baseChunk + min(localBlockId, remainder);
  const int endIdx = startIdx + myChunkSize;

  if (destNode == myNode) {
    // intra node use xgmi for transfer
    for (int idx = warpId; idx < endIdx - startIdx; idx += warpNum) {
      const index_t mapIdx = destPe * MaxNumTokensToRecvPerRank + startIdx + idx;//send到staging的slot号
      size_t mapIdxOffset = mapIdx * stagingOffset;
      const index_t tokenId = args.destPeTokenIdxMap[mapIdx];
      size_t tokenOffset = tokenId * size_t(config.hiddenDim) * sizeof(T);
      const index_t peSortedId = myPe * MaxNumTokensToRecvPerRank + startIdx + idx;//recv staging即shmemInpTokMemObj的写入位置slot号
      size_t peSortedOffset = peSortedId * stagingOffset;
      core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset,
                     reinterpret_cast<char*>(args.inpTokenBuf) + tokenOffset,
                     config.hiddenDim * sizeof(T));
      core::WarpCopy(
          args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset + weightOffset,
          reinterpret_cast<char*>(args.weightsBuf) +
              tokenId * config.numExpertPerToken * sizeof(float),
          config.numExpertPerToken * sizeof(float));
      core::WarpCopy(
          args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset + indicesOffset,
          reinterpret_cast<char*>(args.tokenIndices) +
              tokenId * config.numExpertPerToken * sizeof(index_t),
          config.numExpertPerToken * sizeof(index_t));
      if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
        core::WarpCopy(
            args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset + scalesOffset,
            reinterpret_cast<char*>(args.scalesBuf) +
                tokenId * config.scaleDim * config.scaleTypeSize,
            config.scaleDim * config.scaleTypeSize);
      }
      shmem::ShmemPutTypeNbiWarp<uint8_t>(args.shmemInpTokMemObj, peSortedOffset,
                                          args.shmemStagingTokMemObj, mapIdxOffset, stagingOffset,
                                          destPe);
    }
  } else {
    // inter node use ibgda for transfer
    // last warp for coordinate, other warp for gather token
    __shared__ int gatherTokenNum[1024];
    for (int idx = thdId; idx < 1024; idx += thdNum) {
      gatherTokenNum[idx] = 0;
    }
    __syncthreads();
    const int chunkTokenSize = (warpNum - 1);//要聚合的token数
    if (warpId == warpNum - 1) {//调用ibgda接口
      const int totalTokenInBlock = endIdx - startIdx;
      int chunkOffset = 0;
      int chunkIdx = 0;
      while (chunkOffset < totalTokenInBlock) {
        int actualTokenNum = totalTokenInBlock - chunkOffset < chunkTokenSize
                                 ? totalTokenInBlock - chunkOffset
                                 : chunkTokenSize;
        if (laneId == 0) {
          while (atomicAdd(&gatherTokenNum[chunkIdx], 0) < actualTokenNum) {
            ;  // aggregate 所有warp的token
          }
        }
        // rdma_send
        const index_t srcIdx = destPe * MaxNumTokensToRecvPerRank + startIdx + chunkOffset;
        size_t srcOffset = srcIdx * stagingOffset;
        const index_t dstIdx = myPe * MaxNumTokensToRecvPerRank + startIdx + chunkOffset;
        size_t dstOffset = dstIdx * stagingOffset;
        shmem::ShmemPutTypeNbiWarp<uint8_t>(args.shmemInpTokMemObj, dstOffset,
                                            args.shmemStagingTokMemObj, srcOffset,
                                            actualTokenNum * stagingOffset, destPe);

        ++chunkIdx;
        chunkOffset += chunkTokenSize;
      }
    } else {//负责拷贝到staging
      // int warpTokens = 0;
      int chunkIdx = 0;
      for (int idx = warpId; idx < endIdx - startIdx; idx += chunkTokenSize) {
        const index_t mapIdx = destPe * MaxNumTokensToRecvPerRank + startIdx + idx;
        size_t mapIdxOffset = mapIdx * stagingOffset;
        const index_t tokenId = args.destPeTokenIdxMap[mapIdx];
        size_t tokenOffset = tokenId * size_t(config.hiddenDim) * sizeof(T);
        // const index_t peSortedId = myPe * MaxNumTokensToRecvPerRank + startIdx + idx;
        // size_t peSortedOffset = peSortedId * size_t(config.hiddenDim);
        core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset,
                       reinterpret_cast<char*>(args.inpTokenBuf) + tokenOffset,
                       config.hiddenDim * sizeof(T));
        core::WarpCopy(
            args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset + weightOffset,
            reinterpret_cast<char*>(args.weightsBuf) +
                tokenId * config.numExpertPerToken * sizeof(float),
            config.numExpertPerToken * sizeof(float));
        core::WarpCopy(
            args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset + indicesOffset,
            reinterpret_cast<char*>(args.tokenIndices) +
                tokenId * config.numExpertPerToken * sizeof(index_t),
            config.numExpertPerToken * sizeof(index_t));
        if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
          core::WarpCopy(
              args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset + scalesOffset,
              reinterpret_cast<char*>(args.scalesBuf) +
                  tokenId * config.scaleDim * config.scaleTypeSize,
              config.scaleDim * config.scaleTypeSize);
        }
        if (laneId == 0) atomicAdd(&gatherTokenNum[chunkIdx++], 1);//通知协调warp，一个token已经拷入staging
      }
      // if (laneId == 0 && warpTokens) atomicAdd(&gatherTokenNum, warpTokens);
      __threadfence_block();
    }
  }

  __shared__ index_t recvTokenNum;
  __syncthreads();
  if (thdId == 0) {//同组block都会进来
    // shmem::ShmemAtomicTypeNonFetchWarp<int32_t>(args.recvTokenNumMemObj, myPe * sizeof(index_t),
    //                                         args.shmemStagingTokMemObj->GetMemoryRegion(myPe),
    //                                         myPe * sizeof(index_t), (int32_t)(totalTokens+1),
    //                                         destPe, core::AMO_SET);
    int doneBlockNum = atomicAdd(&args.dispatchGridBarrier[destPe], 1);
    if (doneBlockNum == numsBlockPerDestPe - 1) {// numsBlockPerDestPe个block处理发到同一个destPe，这几个block的同步保证destPe都发完了;最后add的block会进来
      shmem::ShmemPutInt32ImmNbiThread(args.recvTokenNumMemObj, myPe * sizeof(index_t),
                                       totalTokens + 1, destPe);//recvTokenNumMemObj也是个symm buffer，写入的是自己发给destP的token数
      __hip_atomic_store(&args.dispatchGridBarrier[destPe], 0, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);//TODO 同一组的block同步，同样会有bug
    }
  }
  if (thdId == 0) {
    index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>() + destPe;//得到destPe发给自己的token数
    recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
    if (localBlockId == 0) {
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);  // 把所有PE的recvTokenNum加起来，这看着是单机GetDispatchSrcTokenId用的
      args.destPeTokenCounter[destPe] = 0;
    }
    // if (localBlockId == 0) printf("rank[%d] destPe[%d] recvTokenNum: %d\n", myPe, destPe,
    // recvTokenNum);
  }
  __syncthreads();

  const int baseRecvChunk = recvTokenNum / numsBlockPerDestPe;//这里recvTokenNum在sh mem上
  const int recvRemainder = recvTokenNum % numsBlockPerDestPe;
  const int myRecvChunkSize = baseRecvChunk + (localBlockId < recvRemainder);
  // if (localBlockId == 0 && thdId == 0) printf("rank[%d] destPe[%d] myRecvChunkSize: %d\n", myPe,
  // destPe, myRecvChunkSize);
  const int startRecvIdx = localBlockId * baseRecvChunk + min(localBlockId, recvRemainder);
  const int endRecvIdx = startRecvIdx + myRecvChunkSize;
  for (int idx = warpId; idx < myRecvChunkSize; idx += warpNum) {
    index_t localTokenIdx = 0;
    if (laneId == 0) {
      localTokenIdx = atomicAdd(args.localPeTokenCounter, 1);//这里取号又会导致乱序；args.localPeTokenCounter和args.totalRecvTokenNum 真的不是完全相等吗
    }
    localTokenIdx = __shfl(localTokenIdx, 0);
    index_t peSortedId = destPe * MaxNumTokensToRecvPerRank + startRecvIdx + idx;//这里指从destPe收到的token，这个编号就包含了src的信息

    size_t localTokenOffset = size_t(localTokenIdx) * size_t(config.hiddenDim) * sizeof(T);
    size_t peSortedTokenOffset = size_t(peSortedId) * stagingOffset;

    core::WarpCopy(args.shmemOutTokMemObj->template GetAs<char*>() + localTokenOffset,
                   args.shmemInpTokMemObj->template GetAs<char*>() + peSortedTokenOffset,
                   config.hiddenDim * sizeof(T));
    core::WarpCopy(
        args.shmemOutWeightsMemObj->template GetAs<char*>() +
            localTokenIdx * config.numExpertPerToken * sizeof(float),
        args.shmemInpTokMemObj->template GetAs<char*>() + peSortedTokenOffset + weightOffset,
        config.numExpertPerToken * sizeof(float));
    core::WarpCopy(
        args.shmemOutIndicesMemObj->template GetAs<char*>() +
            localTokenIdx * config.numExpertPerToken * sizeof(index_t),
        args.shmemInpTokMemObj->template GetAs<char*>() + peSortedTokenOffset + indicesOffset,
        config.numExpertPerToken * sizeof(index_t));
    if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
      core::WarpCopy(
          args.shmemOutScalesMemObj->template GetAs<char*>() +
              localTokenIdx * config.scaleDim * config.scaleTypeSize,
          args.shmemInpTokMemObj->template GetAs<char*>() + peSortedTokenOffset + scalesOffset,
          config.scaleDim * config.scaleTypeSize);
    }
    if (laneId == 0) {
      args.dispReceiverIdxMap[localTokenIdx] = peSortedId;  // 给用户：destTokId和staging位置；这里有一个坑-peSortedId和前面是不同的，虽然都是startRecvIdx + idx(注意recv)，但send是根据sendTokens划分block的，recv这边是recvToken划分
      args.srcPeTokenIdxMap[peSortedId] = localTokenIdx;//给combine
    }
  }
  SyncIfDebugEnabled("Dispatch kernel: kernel end");
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          BarrierKernel                                         */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ void CrossDeviceBarrierInterNodeKernel(EpDispatchCombineArgs<T> args) {
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = globalThdId / warpSize;

  int warpNum = blockDim.x / warpSize;
  int globalWarpNum = gridDim.x * warpNum;

  if (laneId == 0) atomicAdd(args.combineGridBarrier, 1);

  // TODO: still figure out why use multiple threads lost RDMA writes
  for (int destPe = globalWarpId; destPe < args.config.worldSize; destPe += globalWarpNum) {
    if (laneId == 0) {
      shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, globalWarpNum);
      shmem::ShmemPutUint32ImmNbiWarp(args.crossDeviceBarrierMemObj,
                                      args.config.rank * sizeof(uint32_t),
                                      args.crossDeviceBarrierFlag, destPe);
    }
  }

  uint32_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>();
  if (thdId < args.config.worldSize) {
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId) != args.crossDeviceBarrierFlag) {
    }
  }
  __syncthreads();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpCombineInterNodeKernel                                    */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpCombineInterNodeKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;
  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;
  int globalThdNum = gridDim.x * warpNum * warpSize;

  int myPe = config.rank;
  int npes = config.worldSize;
  int myNode = myPe / MAX_GPUS_PER_NODE;

  size_t MaxNumTokensToSendPerRank = config.MaxNumTokensToSendPerRank();
  size_t MaxNumTokensToRecvPerRank = config.MaxNumTokensToRecvPerRank();

  // Phase 1: send token
  // This phase is symmetric with dispatch recv phase, where tokens are first sent back to its
  // source pe in pe sorted order
  const int numsBlockPerSrcPe = gridDim.x / npes;
  const int srcPe = blockIdx.x / numsBlockPerSrcPe;//几个block处理一个pe
  const int srcNode = srcPe / MAX_GPUS_PER_NODE;
  const int localBlockId = blockIdx.x - srcPe * numsBlockPerSrcPe;
  const int srcPeTokenNum = *(args.recvTokenNumMemObj->template GetAs<index_t*>() + srcPe) - 1;
  const int baseChunk = srcPeTokenNum / numsBlockPerSrcPe;//token再均分给block，应该叫baseChunkToken
  const int remainder = srcPeTokenNum % numsBlockPerSrcPe;//剩下的token？reminder什么鬼命名

  const int myChunkSize = baseChunk + (localBlockId < remainder);//numTokenPerBlock

  const int startIdx = localBlockId * baseChunk + min(localBlockId, remainder);
  const int endIdx = startIdx + myChunkSize;

  const size_t tokenSize = config.hiddenDim * sizeof(T);
  const size_t weightSize = args.weightsBuf ? config.numExpertPerToken * sizeof(float) : 0;
  const size_t tokenPackSize = tokenSize + weightSize;

  if (srcNode == myNode) {
    // intra node use xgmi for transfer
    for (int idx = warpId; idx < endIdx - startIdx; idx += warpNum) {
      const index_t mapIdx = srcPe * MaxNumTokensToRecvPerRank + startIdx + idx;
      size_t mapIdxOffset = mapIdx * tokenPackSize;
      const index_t tokenId = args.srcPeTokenIdxMap[mapIdx];//recv里的token slot更贴切
      size_t tokenOffset = tokenId * tokenSize;
      const index_t peSortedId = myPe * MaxNumTokensToRecvPerRank + startIdx + idx;
      size_t peSortedOffset = peSortedId * tokenPackSize;
      core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset,
                     reinterpret_cast<char*>(args.inpTokenBuf) + tokenOffset, tokenSize);

      if (args.weightsBuf) {
        core::WarpCopy(
            args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset + tokenSize,
            reinterpret_cast<char*>(args.weightsBuf) +
                tokenId * config.numExpertPerToken * sizeof(float),
            weightSize);
      }

      shmem::ShmemPutTypeNbiWarp<uint8_t>(args.shmemInpTokMemObj, peSortedOffset,
                                          args.shmemStagingTokMemObj, mapIdxOffset, tokenPackSize,
                                          srcPe);
    }
  } else {
    // inter node use ibgda for transfer
    // last warp for coordinate, other warp for gather token
    __shared__ int gatherTokenNum[1024];
    for (int idx = thdId; idx < 1024; idx += thdNum) {
      gatherTokenNum[idx] = 0;
    }
    __syncthreads();
    const int chunkTokenSize = (warpNum - 1);
    if (warpId == warpNum - 1) {
      const int totalTokenInBlock = endIdx - startIdx;
      int chunkOffset = 0;
      int chunkIdx = 0;
      while (chunkOffset < totalTokenInBlock) {
        int actualTokenNum = totalTokenInBlock - chunkOffset < chunkTokenSize
                                    ? totalTokenInBlock - chunkOffset
                                    : chunkTokenSize;
        if (laneId == 0) {
          while (atomicAdd(&gatherTokenNum[chunkIdx], 0) < actualTokenNum) {
            ;
          }
        }
        // rdma_send
        const index_t srcIdx = srcPe * MaxNumTokensToRecvPerRank + startIdx + chunkOffset;
        size_t srcOffset = srcIdx * tokenPackSize;
        const index_t dstIdx = myPe * MaxNumTokensToRecvPerRank + startIdx + chunkOffset;
        size_t dstOffset = dstIdx * tokenPackSize;
        shmem::ShmemPutTypeNbiWarp<uint8_t>(args.shmemInpTokMemObj, dstOffset,
                                            args.shmemStagingTokMemObj, srcOffset,
                                            actualTokenNum * tokenPackSize, srcPe);

        ++chunkIdx;
        chunkOffset += chunkTokenSize;
      }
    } else {
      // int warpTokens = 0;
      int chunkIdx = 0;
      for (int idx = warpId; idx < endIdx - startIdx; idx += chunkTokenSize) {
        const index_t mapIdx = srcPe * MaxNumTokensToRecvPerRank + startIdx + idx;
        size_t mapIdxOffset = mapIdx * tokenPackSize;
        const index_t tokenId = args.srcPeTokenIdxMap[mapIdx];
        size_t tokenOffset = tokenId * tokenSize;
        // const index_t peSortedId = myPe * MaxNumTokensToRecvPerRank + startIdx + idx;
        // size_t peSortedOffset = peSortedId * size_t(config.hiddenDim);
        core::WarpCopy(args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset,
                       reinterpret_cast<char*>(args.inpTokenBuf) + tokenOffset, tokenSize);

        if (args.weightsBuf) {
          core::WarpCopy(
              args.shmemStagingTokMemObj->template GetAs<char*>() + mapIdxOffset + tokenSize,
              reinterpret_cast<char*>(args.weightsBuf) +
                  tokenId * config.numExpertPerToken * sizeof(float),
              weightSize);
        }
        if (laneId == 0) atomicAdd(&gatherTokenNum[chunkIdx++], 1);
      }
      // if (laneId == 0 && warpTokens) atomicAdd(&gatherTokenNum, warpTokens);
      __threadfence_block();
    }
  }
  SyncIfDebugEnabled("Combine kernel: send token end");

  // Make sure copy on all GPUs are finished
  CrossDeviceBarrierInterNodeKernel(args);

  if (globalThdId < npes) {
    args.recvTokenNumMemObj->template GetAs<index_t*>()[globalThdId] = 0;
  }

  if (globalThdId == 0) {
    __hip_atomic_store(args.combineGridBarrier, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    args.localPeTokenCounter[0] = 0;
    args.totalRecvTokenNum[0] = 0;
  }

  SyncIfDebugEnabled("Dispatch kernel: sync across device end");

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;

  int warpsPerToken = (globalWarpNum + args.curRankNumToken - 1) / args.curRankNumToken;
  size_t hiddenDimPerWarp = (config.hiddenDim + warpsPerToken - 1) / warpsPerToken;

  for (int i = globalWarpId; i < (args.curRankNumToken * warpsPerToken); i += globalWarpNum) {
    int tokenId = i / warpsPerToken;
    int inTokenPartId = i % warpsPerToken;
    size_t hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
    size_t hiddenDimSize = std::min(config.hiddenDim - hiddenDimOffset, hiddenDimPerWarp);

    // Prepare data pointers on different GPUs
    for (int j = laneId; j < config.numExpertPerToken; j += warpSize) {
      index_t peSortedId = args.dispSenderIdxMap[tokenId * config.numExpertPerToken + j];
      index_t destPe = peSortedId / MaxNumTokensToRecvPerRank;
      size_t byteOffset = size_t(peSortedId) * tokenPackSize + hiddenDimOffset * sizeof(T);
      size_t weightByteOffset = size_t(peSortedId) * tokenPackSize + tokenSize;

      if (destPe < config.worldSize) {
        srcPtrs[j] =
            reinterpret_cast<T*>(args.shmemInpTokMemObj->template GetAs<char*>() + byteOffset);
        srcWeightsPtr[j] = reinterpret_cast<float*>(
            args.shmemInpTokMemObj->template GetAs<char*>() + weightByteOffset);
      } else {
        srcPtrs[j] = nullptr;
        srcWeightsPtr[j] = nullptr;
      }
    }

    size_t offset = size_t(tokenId) * size_t(config.hiddenDim) + hiddenDimOffset;
    core::WarpAccum<T, 8>(args.shmemOutTokMemObj->template GetAs<T*>() + offset, srcPtrs, nullptr,
                          config.numExpertPerToken, hiddenDimSize);

    if (args.weightsBuf && inTokenPartId == warpsPerToken - 1) {
      core::WarpAccum<float, 4>(
          args.shmemOutWeightsMemObj->template GetAs<float*>() + tokenId * config.numExpertPerToken,
          srcWeightsPtr, nullptr, config.numExpertPerToken, config.numExpertPerToken);
    }
  }
}

}  // namespace moe
}  // namespace mori
