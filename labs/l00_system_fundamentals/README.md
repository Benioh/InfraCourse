# L00 · 系统基础：从 CPU 到 GPU 的计算机体系结构

这是整个课程的前置知识补课，覆盖做 GPU 系统工程所需的计算机基础。如果你已经有扎实的 OS/ICS 基础可以快速跳过；如果你是从应用层直接入手的同学，建议花一天时间过一遍。

## 你会学到什么

### 进程与线程
- 进程是资源分配的单位（独立地址空间），线程是调度的单位（共享地址空间）。
- Python 的 GIL：同一时刻只有一个线程执行 Python bytecode。
- `multiprocessing` 绕过 GIL：每个子进程有独立 Python 解释器。
- `torchrun` 本质是 multiprocessing：每个 rank 是一个独立进程。

### 内存层级（CPU 侧）
- L1 Cache (~1ns) → L2 Cache (~5ns) → L3 Cache (~20ns) → DRAM (~100ns)
- Cache line (64 bytes)：CPU 读内存的最小单位。
- 为什么连续内存访问比随机访问快：cache line prefetch。
- 这和 GPU 的 coalesced memory access 是同一个道理。

### Pinned Memory
- 普通内存（pageable）可以被 OS 换出到 disk（swap）。
- Pinned memory（page-locked）保证在物理内存中不被换出。
- CPU→GPU 传输（H2D）：pinned memory 可以走 DMA（不经过 CPU），速度快 2-3×。
- PyTorch 中：`tensor.pin_memory()` 或 DataLoader `pin_memory=True`。
- 代价：占用物理内存，分配较慢。过多 pin 会导致 OOM。

### RDMA / InfiniBand / NVLink
- **PCIe**：CPU↔GPU 通信通道，~32GB/s (Gen4 x16)，~64GB/s (Gen5)。
- **NVLink**：GPU↔GPU 直连，~450GB/s (4th gen, H100)。比 PCIe 快 7-14×。
- **InfiniBand (IB)**：跨节点网络，~50GB/s (HDR)，~100GB/s (NDR)。
- **RDMA**：Remote DMA，网卡直接读写远端内存，绕过 CPU。IB 和 RoCE 都支持 RDMA。
- **GPUDirect RDMA**：网卡直接读写 GPU 显存，不经过 CPU 内存。NCCL 跨节点用这个。

### Python multiprocessing 与 DataLoader
- DataLoader 的 `num_workers > 0`：创建多个子进程做数据预处理。
- 子进程通过 shared memory 或 pipe 把 batch 传回主进程。
- `pin_memory=True`：主进程收到 batch 后 pin 到 pinned memory，加速 H2D。
- `prefetch_factor`：每个 worker 提前准备几个 batch（默认 2）。
- Worker 的随机种子和 rank：需要正确设置避免所有 worker 产出相同数据。

### 文件 I/O 与 mmap
- `mmap`：把文件映射到进程地址空间，读文件变成读内存（OS 按需加载 page）。
- Megatron 的 `.bin` 数据集就用 mmap：支持随机访问，不需要全部加载到 RAM。
- mmap 的好处：多进程可以共享同一份 page cache，节省内存。
- mmap 的代价：首次访问会触发 page fault（磁盘读），随机访问在 HDD 上很慢。

## Quiz

quiz.yaml 覆盖以上所有主题。

## Patch

实现基础系统知识的验证函数：
- 判断一个内存地址是否 page-aligned
- 计算 pinned memory 和 pageable memory 的理论传输时间差
- 根据网络拓扑判断通信瓶颈（PCIe / NVLink / IB）
- 计算 DataLoader 的理论吞吐上限

## 进入下一讲

完成后进入 [L01 环境探针](../l01_env_conda_cuda/README.md)。
