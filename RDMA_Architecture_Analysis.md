# Mori RDMA 架构实现分析

## 概述

Mori 是一个基于 GPU 直接访问 RDMA 的高性能通信库，主要针对 AMD ROCm 平台优化。该库实现了从 GPU kernel 直接发起 RDMA 操作的能力，避免了 CPU 介入，大幅提升了通信性能。

## 核心架构组件

### 1. 传输层架构 (`include/mori/application/transport/`)

#### 传输类型支持
- **RDMA**: 远程直接内存访问，支持跨节点通信
- **P2P**: 同节点内 GPU 之间的点对点通信

```cpp
enum TransportType { RDMA = 0, P2P = 1 };
```

#### RDMA 核心组件

**1. RdmaContext (RDMA 上下文)**
- 管理所有 RDMA 设备的发现和初始化
- 支持 Mellanox MLX5 设备
- 自动检测活跃的设备端口

**2. RdmaDevice (RDMA 设备)**
- 抽象硬件设备接口
- 查询设备属性和端口状态
- 创建设备上下文

**3. RdmaDeviceContext (设备上下文)**
- 管理保护域 (Protection Domain)
- 内存区域注册和管理
- 端点创建和连接

**4. RdmaEndpoint (RDMA 端点)**
- 封装队列对 (QP) 和完成队列 (CQ)
- 包含工作队列句柄和完成队列句柄
- 支持 GPU 直接操作

### 2. 核心传输原语 (`include/mori/core/transport/rdma/`)

#### 设备级原语 (Device Primitives)
这些函数可以在 GPU kernel 中直接调用：

**发送原语**
- `PostSend`: 发送消息
- `PostWrite`: RDMA 写操作
- `PostRead`: RDMA 读操作
- `PostWriteInline`: 内联写操作（小数据）
- `PostAtomic`: 原子操作

**门铃和完成队列操作**
- `UpdateSendDbrRecord`: 更新发送门铃记录
- `RingDoorbell`: 敲门铃通知硬件
- `PollCq`: 轮询完成队列
- `UpdateCqDbrRecord`: 更新完成队列门铃记录

#### MLX5 特定实现
针对 Mellanox MLX5 网卡的优化实现，包括：
- 工作队列元素 (WQE) 的构造
- 硬件特定的门铃机制
- 完成队列元素 (CQE) 的解析

### 3. 应用层接口 (`include/mori/application/`)

#### Context (上下文管理)
- 统一管理 RDMA 和 P2P 传输
- 自动选择最优传输方式
- 处理节点内和节点间通信

#### Memory (内存管理)
- **SymmetricMemory**: 对称内存，所有参与进程都可访问
- **MemoryRegion**: 内存区域，包含地址、本地密钥、远程密钥
- 支持 GPU 内存的 RDMA 注册

#### Bootstrap (引导网络)
- 支持 MPI 和 PyTorch 引导
- 处理端点信息交换
- 建立初始连接

### 4. SHMEM 接口 (`include/mori/shmem/`)

提供类似 OpenSHMEM 的编程接口，但针对 GPU 优化：

#### 核心操作
- **Put 操作**: 将数据写入远程内存
- **Get 操作**: 从远程内存读取数据
- **Atomic 操作**: 原子读写操作
- **同步操作**: 栅栏、等待等

#### 传输类型动态分发
```cpp
#define DISPATCH_TRANSPORT_TYPE(func, pe, ...)                                    \
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();                           \
  application::TransportType transportType = globalGpuStates->transportTypes[pe]; \
  if (transportType == application::TransportType::RDMA) {                        \
    func<application::TransportType::RDMA>(__VA_ARGS__);                          \
  } else if (transportType == application::TransportType::P2P) {                  \
    func<application::TransportType::P2P>(__VA_ARGS__);                           \
  }
```

## 关键特性

### 1. GPU 直接 RDMA (GPU Direct RDMA)
- GPU kernel 可以直接发起 RDMA 操作
- 避免 CPU 参与，减少延迟
- 支持从 GPU 直接访问远程 GPU 内存

### 2. 零拷贝通信
- 直接在 GPU 内存间传输数据
- 避免主机内存中转
- 最大化带宽利用率

### 3. 异步操作支持
- 非阻塞的 RDMA 操作
- 支持多操作并发
- 完成队列轮询机制

### 4. 多设备支持
- 自动检测和使用多个 RDMA 设备
- 负载均衡和容错机制
- 灵活的设备选择策略

## 典型使用流程

### 1. 初始化阶段
```cpp
// 1. 创建 RDMA 上下文
RdmaContext rdmaContext;
RdmaDeviceList devices = rdmaContext.GetRdmaDeviceList();

// 2. 选择设备并创建设备上下文
RdmaDevice* device = devices[0];
RdmaDeviceContext* deviceContext = device->CreateRdmaDeviceContext();

// 3. 创建端点
RdmaEndpointConfig config;
RdmaEndpoint endpoint = deviceContext->CreateRdmaEndpoint(config);
```

### 2. 内存注册
```cpp
// 在 GPU 上分配内存
void* buffer;
hipMalloc(&buffer, size);

// 注册 RDMA 内存区域
MemoryRegion mr = deviceContext->RegisterMemoryRegion(buffer, size, ACCESS_FLAGS);
```

### 3. 连接建立
```cpp
// 交换端点信息（通过 MPI 等）
bootNet.Allgather(&endpoint.handle, globalHandles.data(), sizeof(RdmaEndpointHandle));

// 连接到远程端点
deviceContext->ConnectEndpoint(localHandle, remoteHandle);
```

### 4. GPU Kernel 中的 RDMA 操作
```cpp
__global__ void WriteKernel(RdmaEndpoint* endpoint, MemoryRegion localMr, 
                            MemoryRegion remoteMr, size_t size) {
    // 发起 RDMA 写操作
    uint64_t dbrVal = PostWrite<ProviderType::MLX5>(
        endpoint->wqHandle.sqAddr, endpoint->wqHandle.sqWqeNum, 
        &endpoint->wqHandle.postIdx, endpoint->handle.qpn,
        localMr.addr, localMr.lkey, remoteMr.addr, remoteMr.rkey, size);
    
    // 更新门铃记录
    UpdateSendDbrRecord<ProviderType::MLX5>(endpoint->wqHandle.dbrRecAddr,
                                            endpoint->wqHandle.postIdx);
    
    // 敲门铃通知硬件
    RingDoorbell<ProviderType::MLX5>(endpoint->wqHandle.dbrAddr, dbrVal);
    
    // 轮询完成
    int opcode = PollCq<ProviderType::MLX5>(endpoint->cqHandle.cqAddr, 
                                            endpoint->cqHandle.cqeSize,
                                            endpoint->cqHandle.cqeNum, 
                                            &endpoint->cqHandle.consIdx);
}
```

## 性能优化策略

### 1. 内存对齐和布局优化
- 4KB 页面对齐
- 缓存行优化的数据结构
- GPU 内存合并访问

### 2. 批处理操作
- 支持多个 WQE 批量提交
- 减少门铃开销
- 提高吞吐量

### 3. 锁优化
- 原子操作替代锁
- 无锁队列操作
- 减少线程同步开销

### 4. 硬件特性利用
- MLX5 硬件特定优化
- 内联数据传输
- 完成队列批量处理

## 应用场景

1. **分布式深度学习**
   - 梯度聚合和参数同步
   - 模型并行训练
   - 大规模训练加速

2. **高性能计算**
   - 科学计算应用
   - 矩阵运算加速
   - 数值模拟

3. **数据分析**
   - 大数据处理
   - 图计算
   - 实时分析

## 总结

Mori RDMA 实现提供了一个完整的 GPU 直接 RDMA 通信栈，从底层硬件抽象到高级编程接口都有良好的封装。其核心优势在于：

1. **低延迟**: GPU 直接操作，避免 CPU 介入
2. **高带宽**: 零拷贝数据传输
3. **易用性**: 类似 SHMEM 的编程接口
4. **可扩展性**: 支持多设备和大规模集群
5. **跨平台**: 支持不同的 RDMA 硬件

这个实现为构建高性能的分布式 GPU 应用提供了强大的基础设施。

podman run -d \
    --name dev_primus \
    --ipc=host \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/infiniband \
    --cap-add=SYS_PTRACE \
    --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --group-add video \
    --privileged \
    -v  ~/:/workspace \
     docker.io/rocm/megatron-lm:v25.5_py310 sleep infinity