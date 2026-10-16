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

// KfdPinCheck must predict how far hipMemGetInfo is off for a registered VRAM
// allocation, for every mix of dma-buf and peer-memory MRs and in either order
// of deregistration, on drivers that miscount RDMA pins (amdgpu-dkms 6.14/6.16,
// ROCm/mori#721) and on drivers that do not.
//
// Needs a GPU and two RDMA devices; pick the devices with MORI_RDMA_DEVICES.
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <stdlib.h>

#include <memory>
#include <vector>

#include "mori/application/transport/rdma/rdma.hpp"
#include "src/application/transport/rdma/kfd_pin_check.hpp"

using namespace mori::application;

namespace {

constexpr size_t kBufferSize = 4ull << 30;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
// hipMemGetInfo moves in 2 MiB steps, and the HIP runtime makes small
// allocations of its own.
constexpr double kSlack = 64.0 * 1024 * 1024;

double FreeVram() {
  size_t free = 0, total = 0;
  EXPECT_EQ(hipMemGetInfo(&free, &total), hipSuccess);
  return static_cast<double>(free);
}

std::unique_ptr<RdmaDeviceContext> NewContext(RdmaDevice* device, bool dmabufFirst) {
  if (dmabufFirst) setenv("MORI_ENABLE_DMABUF_REG", "1", 1);
  std::unique_ptr<RdmaDeviceContext> ctx(device->CreateRdmaDeviceContext());
  if (dmabufFirst) unsetenv("MORI_ENABLE_DMABUF_REG");
  return ctx;
}

class KfdPinCheckTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    context = std::make_unique<RdmaContext>(RdmaBackendType::IBVerbs);
    for (const ActiveDevicePort& port : GetActiveDevicePortList(context->GetRdmaDeviceList())) {
      if (devices.empty() || devices.back() != port.first) devices.push_back(port.first);
    }
    if (devices.size() < 2) GTEST_SKIP() << "needs two active RDMA devices";
    ASSERT_EQ(hipMalloc(&buf, kBufferSize), hipSuccess);
    allocated = FreeVram();
  }

  void TearDown() override {
    if (buf != nullptr) EXPECT_EQ(hipFree(buf), hipSuccess);
  }

  // Fails the test when the check did not calibrate, e.g. with MORI_KFD_PIN_CHECK=0.
  bool Calibrated() {
    if (KfdPinCheck::Instance().Calibrated()) return true;
    ADD_FAILURE() << "KfdPinCheck did not calibrate";
    return false;
  }

  void ExpectDiagnosed(const char* step) {
    const double measured = FreeVram() - allocated;
    const double diagnosed = static_cast<double>(KfdPinCheck::Instance().FreeVramError(buf));
    EXPECT_NEAR(measured, diagnosed, kSlack)
        << step << ": hipMemGetInfo is off by " << measured / kGiB << " GiB, check says "
        << diagnosed / kGiB << " GiB";
  }

  std::unique_ptr<RdmaContext> context;
  std::vector<RdmaDevice*> devices;
  void* buf = nullptr;
  double allocated = 0;
};

TEST_F(KfdPinCheckTest, OnePeerMemoryMr) {
  auto a = NewContext(devices[0], false);
  a->RegisterRdmaMemoryRegionAuto(buf, kBufferSize);
  if (!Calibrated()) return;
  ExpectDiagnosed("registered");
  a->DeregisterRdmaMemoryRegion(buf);
  ExpectDiagnosed("deregistered");
}

TEST_F(KfdPinCheckTest, DmabufMr) {
  auto a = NewContext(devices[0], /*dmabufFirst=*/true);
  a->RegisterRdmaMemoryRegionAuto(buf, kBufferSize);
  if (!Calibrated()) return;
  ExpectDiagnosed("registered through dma-buf");
  a->DeregisterRdmaMemoryRegion(buf);
  ExpectDiagnosed("deregistered");
}

TEST_F(KfdPinCheckTest, TwoNics) {
  auto a = NewContext(devices[0], false);
  auto b = NewContext(devices[1], false);
  a->RegisterRdmaMemoryRegionAuto(buf, kBufferSize);
  b->RegisterRdmaMemoryRegionAuto(buf, kBufferSize);
  if (!Calibrated()) return;
  ExpectDiagnosed("registered on two NICs");
  a.reset();
  ExpectDiagnosed("first context destroyed");
  b->DeregisterRdmaMemoryRegion(buf);
  ExpectDiagnosed("both deregistered");
}

TEST_F(KfdPinCheckTest, PeerMemoryAndDmabufMrs) {
  auto a = NewContext(devices[0], false);
  auto b = NewContext(devices[1], /*dmabufFirst=*/true);
  a->RegisterRdmaMemoryRegionAuto(buf, kBufferSize);
  b->RegisterRdmaMemoryRegionAuto(buf, kBufferSize);
  if (!Calibrated()) return;
  ExpectDiagnosed("one MR of each kind");
  a->DeregisterRdmaMemoryRegion(buf);
  ExpectDiagnosed("peer-memory MR deregistered first");
  b->DeregisterRdmaMemoryRegion(buf);
  ExpectDiagnosed("both deregistered");
}

TEST_F(KfdPinCheckTest, SubBuffersOfOneAllocation) {
  auto a = NewContext(devices[0], false);
  char* hi = static_cast<char*>(buf) + kBufferSize / 2;
  a->RegisterRdmaMemoryRegionAuto(buf, kBufferSize / 2);
  a->RegisterRdmaMemoryRegion(hi, kBufferSize / 2);
  if (!Calibrated()) return;
  ExpectDiagnosed("two halves registered");
  a->DeregisterRdmaMemoryRegion(buf);
  ExpectDiagnosed("lower half deregistered");
  a->DeregisterRdmaMemoryRegion(hi);
  ExpectDiagnosed("both deregistered");
}

}  // namespace
