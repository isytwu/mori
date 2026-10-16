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
#include "src/application/transport/rdma/kfd_pin_check.hpp"

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <unistd.h>

#include <cmath>
#include <fstream>
#include <optional>

#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/utils/env_utils.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace application {

namespace {

constexpr size_t kMinTrackedBytes = 64ull << 20;
constexpr size_t kCalibrationBytes = 64ull << 20;
constexpr int kNoise = 2;

std::string AmdgpuVersion() {
  std::ifstream f("/sys/module/amdgpu/version");
  std::string version;
  if (!(f >> version)) version = "in-tree";
  return version;
}

// The VRAM allocation of this process that contains [ptr, ptr + size).
bool FindVramAllocation(void* ptr, size_t size, uintptr_t* base, size_t* allocSize, int* device) {
  hsa_amd_pointer_info_t info{};
  info.size = sizeof(info);
  if (hsa_amd_pointer_info(ptr, &info, nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS ||
      info.type != HSA_EXT_POINTER_TYPE_HSA) {
    return false;
  }
  hsa_device_type_t type;
  if (hsa_agent_get_info(info.agentOwner, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS ||
      type != HSA_DEVICE_TYPE_GPU) {
    return false;
  }
  const uintptr_t begin = reinterpret_cast<uintptr_t>(info.agentBaseAddress);
  const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
  if (p < begin || p + size > begin + info.sizeInBytes) return false;

  hipPointerAttribute_t attr{};
  if (hipPointerGetAttributes(&attr, ptr) != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  *base = begin;
  *allocSize = info.sizeInBytes;
  *device = attr.device;
  return true;
}

double FreeVram() {
  size_t freeBytes = 0, total = 0;
  if (hipMemGetInfo(&freeBytes, &total) != hipSuccess) (void)hipGetLastError();
  return static_cast<double>(freeBytes);
}

// How far free VRAM moved, in calibration allocations: 0, +1 or -1, or kNoise
// when it moved by anything else, i.e. someone allocated or freed meanwhile.
int AllocationSteps(double delta) {
  const double unit = static_cast<double>(kCalibrationBytes);
  for (int steps : {0, 1, -1}) {
    if (std::fabs(delta - steps * unit) < unit / 4) return steps;
  }
  return kNoise;
}

}  // namespace

KfdPinCheck& KfdPinCheck::Instance() {
  static KfdPinCheck* instance = new KfdPinCheck();
  return *instance;
}

KfdPinCheck::KfdPinCheck() {
  std::optional<std::string> value = env::GetString("MORI_KFD_PIN_CHECK");
  if (value.has_value()) enabled = env::detail::ParseBool(value->c_str()).value_or(true);
}

void KfdPinCheck::Calibrate(ibv_pd* pd, int device) {
  calibrationTried = true;
  int prevDevice = -1;
  if (hipGetDevice(&prevDevice) != hipSuccess) prevDevice = -1;
  void* buf = nullptr;
  if (hipSetDevice(device) != hipSuccess || hipMalloc(&buf, kCalibrationBytes) != hipSuccess) {
    (void)hipGetLastError();
    if (prevDevice >= 0) (void)hipSetDevice(prevDevice);
    MORI_APP_DEBUG("KFD pin check: no {} MiB on GPU {} to calibrate with; check disabled",
                   kCalibrationBytes >> 20, device);
    return;
  }

  // Page-sized MRs still pin the whole allocation, so each pin moves free VRAM
  // by 0 or one calibration allocation.
  const size_t page = static_cast<size_t>(getpagesize());
  for (int attempt = 0; attempt < 2 && !calibrated; ++attempt) {
    double before = FreeVram();
    uint64_t offset = 0;
    int fd = TryExportDmabufFd(buf, page, &offset);
    ibv_mr* viaDmabuf = fd >= 0
                            ? ibv_reg_dmabuf_mr(pd, offset, page, reinterpret_cast<uint64_t>(buf),
                                                fd, IBV_ACCESS_LOCAL_WRITE)
                            : nullptr;
    if (fd >= 0) close(fd);
    const int dmabuf = viaDmabuf ? AllocationSteps(FreeVram() - before) : 0;
    if (viaDmabuf) ibv_dereg_mr(viaDmabuf);

    before = FreeVram();
    ibv_mr* first = ibv_reg_mr(pd, buf, page, IBV_ACCESS_LOCAL_WRITE);
    const int one = first ? AllocationSteps(FreeVram() - before) : 0;
    ibv_mr* second = first ? ibv_reg_mr(pd, buf, page, IBV_ACCESS_LOCAL_WRITE) : nullptr;
    const int two = second ? AllocationSteps(FreeVram() - before) : one;
    if (second) ibv_dereg_mr(second);
    if (first) ibv_dereg_mr(first);

    if ((dmabuf == 0 || dmabuf == -1) && (one == 0 || one == -1) &&
        (two - one == 0 || two - one == 1)) {
      calibrated = true;
      peerMemory = first != nullptr;
      dmabufPin = dmabuf;
      firstPeerPin = one;
      extraPeerPin = two - one;
    }
  }
  (void)hipFree(buf);
  if (prevDevice >= 0) (void)hipSetDevice(prevDevice);

  driver = AmdgpuVersion();
  if (calibrated) {
    MORI_APP_DEBUG(
        "KFD pin check: amdgpu {}, peer memory {}; one pin moves free VRAM by {} (dma-buf), {} "
        "(first peer-memory MR), {} (each further one) allocations",
        driver, peerMemory ? "available" : "unavailable", dmabufPin, firstPeerPin, extraPeerPin);
  } else {
    MORI_APP_DEBUG("KFD pin check: free VRAM changed during calibration; check disabled");
  }
}

int64_t KfdPinCheck::ErrorOf(const Allocation& a) const {
  if (!calibrated) return 0;
  const int64_t size = static_cast<int64_t>(a.size);
  if (a.peerMrs > 0) return (firstPeerPin + (a.peerMrs - 1) * extraPeerPin) * size;
  if (a.dmabufMrs > 0) return dmabufPin * size;
  return 0;
}

// Logged as errors: mori shows only errors by default, and the wrong count makes
// later allocations fail.
void KfdPinCheck::ReportIfWrong(uintptr_t base, const Allocation& a) {
  const int64_t error = ErrorOf(a);
  const size_t mib = a.size >> 20;
  if (error < 0) {
    if (reportedLow) {
      MORI_APP_DEBUG("KFD pin check: VRAM allocation {:#x} ({} MiB) is also counted twice", base,
                     mib);
      return;
    }
    reportedLow = true;
    if (a.peerMrs > 0) {
      MORI_APP_ERROR(
          "KFD pin check: amdgpu {} counts RDMA pins of VRAM twice (ROCm/mori#721): while VRAM "
          "allocation {:#x} ({} MiB) is registered, hipMemGetInfo and new allocations see {} MiB "
          "less free VRAM. Install amdgpu-dkms 31.40.1.1 or later, or leave that much headroom.",
          driver, base, mib, mib);
      return;
    }
    const char* reason = a.dmabufPath == Path::DmabufPreferred
                             ? " because MORI_ENABLE_DMABUF_REG is set"
                         : a.dmabufPath == Path::DmabufFallback ? " after ibv_reg_mr failed for it"
                                                                : "";
    const char* advice =
        !peerMemory
            ? "This host has no peer memory for ibv_reg_mr, so install amdgpu-dkms 31.40.1.1 or "
              "later, or leave that much headroom."
        : a.dmabufPath == Path::DmabufPreferred
            ? "ibv_reg_mr works on this host: unset MORI_ENABLE_DMABUF_REG, or install "
              "amdgpu-dkms 31.40.1.1 or later."
            : "Install amdgpu-dkms 31.40.1.1 or later, or leave that much headroom.";
    MORI_APP_ERROR(
        "KFD pin check: VRAM allocation {:#x} ({} MiB) is registered for RDMA through dma-buf{}, "
        "and amdgpu {} counts dma-buf pins of VRAM twice (ROCm/mori#721): hipMemGetInfo and new "
        "allocations see {} MiB less free VRAM until it is deregistered. {}",
        base, mib, reason, driver, mib, advice);
  } else if (error > 0) {
    if (reportedHigh) {
      MORI_APP_DEBUG("KFD pin check: VRAM allocation {:#x} ({} MiB) is also over-reported", base,
                     mib);
      return;
    }
    reportedHigh = true;
    MORI_APP_ERROR(
        "KFD pin check: VRAM allocation {:#x} ({} MiB) has {} ibv_reg_mr MRs (several NICs or "
        "sub-ranges), and amdgpu {} adds every peer-memory pin back (ROCm/mori#721): "
        "hipMemGetInfo reports {} MiB more free VRAM than there is, and allocations sized from "
        "it can fail. Install amdgpu-dkms 31.40.1.1 or later.",
        base, mib, a.peerMrs, driver, error >> 20);
  }
}

void KfdPinCheck::OnRegistered(ibv_pd* pd, void* ptr, size_t size, ibv_mr* mr, Path path) {
  if (!enabled) return;
  uintptr_t base = 0;
  size_t allocSize = 0;
  int device = -1;
  if (!FindVramAllocation(ptr, size, &base, &allocSize, &device) || allocSize < kMinTrackedBytes) {
    return;
  }
  std::lock_guard<std::mutex> lock(mu);
  if (!calibrationTried) Calibrate(pd, device);
  if (!calibrated || (dmabufPin == 0 && firstPeerPin == 0 && extraPeerPin == 0)) return;

  Allocation& a = allocations[base];
  a.size = allocSize;
  const bool peer = path == Path::PeerMemory;
  if (peer) {
    a.peerMrs++;
  } else {
    a.dmabufMrs++;
    a.dmabufPath = path;
  }
  mrs[mr] = {base, peer};
  ReportIfWrong(base, a);
}

void KfdPinCheck::OnDeregistering(ibv_mr* mr) {
  if (!enabled) return;
  std::lock_guard<std::mutex> lock(mu);
  auto it = mrs.find(mr);
  if (it == mrs.end()) return;
  const Mr gone = it->second;
  mrs.erase(it);
  auto owner = allocations.find(gone.base);
  if (owner == allocations.end()) return;
  Allocation& a = owner->second;
  if (gone.peer) {
    a.peerMrs--;
  } else {
    a.dmabufMrs--;
  }
  if (a.peerMrs == 0 && a.dmabufMrs == 0) {
    allocations.erase(owner);
    return;
  }
  ReportIfWrong(gone.base, a);
}

int64_t KfdPinCheck::FreeVramError(void* ptr) {
  uintptr_t base = 0;
  size_t allocSize = 0;
  int device = -1;
  if (!FindVramAllocation(ptr, 1, &base, &allocSize, &device)) return 0;
  std::lock_guard<std::mutex> lock(mu);
  auto it = allocations.find(base);
  return it == allocations.end() ? 0 : ErrorOf(it->second);
}

bool KfdPinCheck::Calibrated() {
  std::lock_guard<std::mutex> lock(mu);
  return calibrated;
}

}  // namespace application
}  // namespace mori
