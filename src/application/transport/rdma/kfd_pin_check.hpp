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

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include "infiniband/verbs.h"

namespace mori {
namespace application {

// amdgpu-dkms 6.14.x and 6.16.x miscount RDMA pins of KFD VRAM in the free
// memory that hipMemGetInfo reports and new allocations are checked against
// (fixed in amdgpu-dkms 6.19.14, AMD Linux release 31.40.1.1; ROCm/mori#721):
//  - a dma-buf pin (ibv_reg_dmabuf_mr) subtracts the whole allocation a second
//    time, so later allocations and kernel loads can fail with VRAM to spare;
//  - every peer-memory pin (ibv_reg_mr) adds the whole allocation back, so N
//    such MRs on one allocation over-report free VRAM N-1 times.
// KfdPinCheck measures once how the driver counts these pins, tracks the MRs
// of each VRAM allocation of 64 MiB or more, and logs an error the first time
// they make free VRAM wrong. It does not change how MRs are registered.
// MORI_KFD_PIN_CHECK=0 turns it off.
class KfdPinCheck {
 public:
  enum class Path { PeerMemory, Dmabuf, DmabufPreferred, DmabufFallback };

  static KfdPinCheck& Instance();

  void OnRegistered(ibv_pd* pd, void* ptr, size_t size, ibv_mr* mr, Path path);
  void OnDeregistering(ibv_mr* mr);

  // How far free VRAM is off right now because of the MRs on the allocation
  // containing `ptr`: negative when too little is reported, positive when too
  // much. Zero if the allocation is untracked or the driver was not measured.
  int64_t FreeVramError(void* ptr);
  bool Calibrated();

 private:
  struct Allocation {
    size_t size{0};
    int peerMrs{0};
    int dmabufMrs{0};
    Path dmabufPath{Path::Dmabuf};
  };
  struct Mr {
    uintptr_t base;
    bool peer;
  };

  KfdPinCheck();
  void Calibrate(ibv_pd* pd, int device);
  int64_t ErrorOf(const Allocation& a) const;
  void ReportIfWrong(uintptr_t base, const Allocation& a);

  std::mutex mu;
  bool enabled{true};
  bool calibrationTried{false};
  bool calibrated{false};
  bool peerMemory{false};
  // Effect of one pin on free VRAM, in allocations: a dma-buf pin and the first
  // peer-memory pin are 0 or -1, each further peer-memory pin is 0 or +1.
  int dmabufPin{0};
  int firstPeerPin{0};
  int extraPeerPin{0};
  std::string driver;
  bool reportedLow{false};
  bool reportedHigh{false};
  std::unordered_map<uintptr_t, Allocation> allocations;
  std::unordered_map<ibv_mr*, Mr> mrs;
};

}  // namespace application
}  // namespace mori
