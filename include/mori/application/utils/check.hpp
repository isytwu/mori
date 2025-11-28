#pragma once

#include <execinfo.h>
#include <hip/hip_runtime.h>
#include <unistd.h>

#include "rocm_smi/rocm_smi.h"

namespace mori {
namespace application {

#define HIP_RUNTIME_CHECK(stmt)                                            \
  do {                                                                     \
    hipError_t result = (stmt);                                            \
    if (hipSuccess != result) {                                            \
      fprintf(stderr, "[%s:%d] hip failed with %s \n", __FILE__, __LINE__, \
              hipGetErrorString(result));                                  \
      exit(-1);                                                            \
    }                                                                      \
    assert(hipSuccess == result);                                          \
  } while (0)

#define HIP_RUNTIME_CHECK_WITH_BACKTRACE(stmt)                             \
  do {                                                                     \
    hipError_t result = (stmt);                                            \
    if (hipSuccess != result) {                                            \
      fprintf(stderr, "[%s:%d] hip failed with %s \n", __FILE__, __LINE__, \
              hipGetErrorString(result));                                  \
      void* array[20];                                                     \
      int size = backtrace(array, 20);                                     \
      backtrace_symbols_fd(array, size, STDERR_FILENO);                    \
      exit(-1);                                                            \
    }                                                                      \
    assert(hipSuccess == result);                                          \
  } while (0)

#define SYSCALL_RETURN_ZERO(stmt)                                                               \
  do {                                                                                          \
    auto _ret = (stmt);                                                                         \
    if (_ret != 0) {                                                                            \
      fprintf(stderr, "[%s:%d] syscall failed with %s\n", __FILE__, __LINE__, strerror(errno)); \
      exit(-1);                                                                                 \
    }                                                                                           \
  } while (0)

#define SYSCALL_RETURN_ZERO_IGNORE_ERROR(stmt, ignored)                                           \
  do {                                                                                            \
    auto _ret = (stmt);                                                                           \
    if (_ret != 0) {                                                                              \
      int err = errno;                                                                            \
      if (err != ignored) {                                                                       \
        fprintf(stderr, "[%s:%d] syscall failed with %s\n", __FILE__, __LINE__, strerror(errno)); \
        exit(-1);                                                                                 \
      }                                                                                           \
    }                                                                                             \
  } while (0)

#define ROCM_SMI_CHECK(stmt)                                                          \
  do {                                                                                \
    rsmi_status_t result = (stmt);                                                    \
    if (RSMI_STATUS_SUCCESS != result) {                                              \
      const char* msg;                                                                \
      rsmi_status_string(result, &msg);                                               \
      fprintf(stderr, "[%s:%d] rocm smi failed with %s \n", __FILE__, __LINE__, msg); \
      exit(-1);                                                                       \
    }                                                                                 \
    assert(RSMI_STATUS_SUCCESS == result);                                            \
  } while (0)

}  // namespace application

inline hipError_t HipMallocWithLog(void** ptr, size_t size, const char* file = __FILE__,
                                   int line = __LINE__) {
  printf("[hipMalloc] %s:%d size=%zu bytes\n", file, line, size);
  return hipMalloc(ptr, size);
}

}  // namespace mori