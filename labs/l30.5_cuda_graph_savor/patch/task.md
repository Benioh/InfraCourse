# L10.7 Patch · CUDA Graph Cache + Memory Savor (CPU 模拟)

## 你要交付什么

两个 primitive：

```python
class GraphCache:
    def capture_or_replay(
        self,
        bs: int,
        forward_fn: Callable[[Tensor], Tensor],
        real_input: Tensor,
    ) -> Tensor:
        """按 bs 缓存 captured graph；hit 时复用 static input_buffer (copy_)
        + 只调一次 forward_fn 写入 output_buffer，返回 output 的 clone。"""

class MemorySavor:
    def register(self, name: str, tensor: Tensor) -> None: ...
    def pause(self) -> None:
        """释放所有 registered tensor 的物理 bytes，记元数据 (shape, dtype, init_value)."""
    def resume(self) -> None:
        """按元数据重建 tensor，恢复物理 bytes."""
    def get(self, name: str) -> Tensor: ...
    def physical_bytes(self) -> int: ...
```

**禁止** import `torch.cuda.graph` 真实 API（CPU 没有）。
**允许** torch.zeros / clone / data_ptr。

补丁规模目标：80–120 行。

## 关键设计点（这是真实工程的精髓）

1. **CUDA Graph 的核心** = static buffers + 录制好的 kernel 序列。replay 时只需
   把新输入 copy 到静态 input buffer，graph 会读静态 buffer、写静态 output buffer。
   **不能** 每次 replay 都 alloc 新 tensor，否则就退化成普通 forward 了。

2. **Memory Savor 的核心** = CUDA Virtual Memory 让物理页可以"暂停":
   - `pause()`：unmap 物理页，但保留虚拟地址范围 + 元数据
   - `resume()`：重新 map 物理页，按元数据复原 tensor
   CPU 模拟里我们用 dict 替代 VM mapping，行为不变量保持。

3. **协作不变量**：`pause` 后 `physical_bytes() == 0`；`resume` 后恢复成 pause 前的总量。

## 接口契约

```python
import torch

cache = GraphCache()
def fwd(x): return x * 2 + 1

real_input = torch.tensor([1.0, 2.0, 3.0])
out1 = cache.capture_or_replay(bs=3, forward_fn=fwd, real_input=real_input)
assert torch.allclose(out1, torch.tensor([3.0, 5.0, 7.0]))

# 第二次同 bs：static buffer 复用
real_input2 = torch.tensor([10.0, 20.0, 30.0])
out2 = cache.capture_or_replay(bs=3, forward_fn=fwd, real_input=real_input2)
assert torch.allclose(out2, torch.tensor([21.0, 41.0, 61.0]))
assert cache.captures == 1   # 只 capture 一次
assert cache.replays == 1    # 第二次是 replay
```

## 不变量

1. **GraphCache**：
   - 同 bs 第二次调用必须 hit cache（`replays += 1`，`captures` 不变）
   - 不同 bs 触发新 capture
   - 多次 replay 时 `_graphs[bs].input_buffer.data_ptr()` **不变**（buffer 复用）
   - replay 输出与 forward_fn 直接调用结果数值相等
2. **MemorySavor**：
   - `register(name, tensor)` 后 `get(name)` 返回该 tensor
   - `pause()` → `physical_bytes() == 0`
   - `resume()` → `get(name).shape / .dtype` 与 pause 前一致
   - 不允许在 pause 状态下 `get`（抛 RuntimeError）

## 怎么验证

```bash
make patch-test M=l30.5_cuda_graph_savor
```

## 写完之后你能做什么

- 解释 `torch.cuda.graph` capture/replay 的工程价值（消除 CPU launch 开销，对
  decode 一次 1 token 的小工作量尤其重要）。
- 解释为什么 RL co-locate 必须用 memory savor 而不是 cudaFree（前者保留虚拟
  地址，可以"原地"复活；后者会让指针失效，graph 就废了）。
- 看懂 SGLang Dual AR omni 模型用 CUDA Graph + 多图复用统一覆盖的优化。
- 在 Capstone Stage B 给 SGLang 服务接 graph cache，Stage C 给 RL 接 memory savor。
