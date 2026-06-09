# L03 Memory Snapshot Debug Checklist

这份 checklist 用于排查训练、rollout 或推理服务中的长期显存上涨。它假设你已经能跑通最小 snapshot 或等价的 memory history 工具。

## 1. 先确认 OOM 类型

- [ ] OOM 是否发生在第一步或前几步。
- [ ] 显存是否随 step 呈单调或阶梯式增长。
- [ ] batch size、sequence length、dtype、optimizer、activation checkpointing 是否近期变化。
- [ ] `allocated`、`reserved`、`peak` 的口径是否分清。
- [ ] 是否有 hook、cache、debug list、request state 或 closure 新增。

判断：

- 第一轮就爆：先回到 L02 静态账本和 activation peak。
- 跑一段时间后爆：继续做 snapshot 归因。

## 2. 选对进程和窗口

- [ ] 显存上涨发生在哪张 GPU。
- [ ] 对应进程、rank、worker、server pid 是否确认。
- [ ] memory history 是否在泄露窗口前打开。
- [ ] dump 是否发生在 live bytes 已经上涨之后。
- [ ] 多进程系统是否同时检查训练 worker、rollout worker、serving engine 和数据预处理进程。

常见错误：

- 只 dump rank0，但泄露发生在 rollout worker。
- OOM 后进程已退出，snapshot 文件没有落盘。
- 录制窗口太短，没有覆盖泄露分配。

## 3. 看 snapshot 主字段

- [ ] `events` 是否包含 alloc 和 free。
- [ ] `live_allocations` 是否随 step 增长。
- [ ] `total_leaked_bytes` 是否和显存增长量同方向。
- [ ] top stack 的累计 bytes 是否明显高于其他 stack。
- [ ] top stack 是否来自业务代码、hook、cache 或框架内部长期对象。

不要把历史 alloc 总量当作泄露。已释放对象只能帮助理解流量，泄露候选来自 live allocations。

## 4. 回到源码路径

- [ ] top stack 的最内层函数是否能定位到文件和行号。
- [ ] 该函数是否保存 GPU tensor 到长生命周期容器。
- [ ] hook 是否按生命周期 remove。
- [ ] cache 是否在 request finished、episode done 或 epoch end 释放。
- [ ] closure 是否捕获了 tensor，而实际只需要 scalar、shape 或 CPU copy。
- [ ] debug log 是否保存了 tensor 对象，而不是 `.item()` 或摘要。

修复方向：

- 保存标量或 CPU copy。
- 使用 weak reference 或显式 clear。
- 在 finished / abort / exception 路径都释放状态。
- 缩短 hook 和 closure 的作用域。

## 5. 验证修复

- [ ] 使用同一命令、同一配置、同一输入规模重跑。
- [ ] 记录修复前后的 step 区间。
- [ ] 对比 `total_leaked_bytes` 或真实 allocated 曲线。
- [ ] top stack 是否从高位消失，或 bytes 不再随 step 增长。
- [ ] 记录修复代价，例如额外 CPU copy、cache hit 下降或吞吐变化。

## 6. 结束条件

复盘结束时应能写清：

- 发生问题的 rank/worker/pid。
- 增长窗口和触发 OOM 的 step。
- snapshot 文件路径。
- top-k stack 和每项 bytes。
- 哪条引用生命周期不符合预期。
- 修复动作和复测结果。
