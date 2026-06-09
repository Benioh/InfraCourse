# L03.5 System Map · Debug 方法论

## 在全课程中的位置

```
L03 Manual DDP（梯度同步基础）
    ↓
→ L03.5 Debug 方法论 ←（你在这里）
    ↓
L04 GPU Kernel / Triton
    ↓
... 后续所有 Lab 遇到问题时回到这里的方法论
```

## 为什么放在这里

- 学完 L03 后你已经接触了多卡场景，开始可能遇到分布式 bug。
- 后面的 Lab（TP、FSDP、PP、serving、RL）复杂度急剧上升，没有系统的 debug 方法会事倍功半。
- 本讲建立的方法论贯穿整个课程后半段。

## 本讲不做什么

- 不深入某个具体框架的 debug 细节（各 Lab 自行覆盖）
- 不讲 profiler 工具的使用（那是 L02.7）
- 不实际修复某个 bug（本讲是方法论，不是 bug fix）
