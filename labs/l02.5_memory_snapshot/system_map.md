# 系统地图：L03 Memory Snapshot 泄露归因

L03 接在 L02 PyTorch 显存账本之后。L02 先把参数、梯度、optimizer state 和 activation 的正常账本讲清；L03 处理另一类更难定位的问题：显存总量随 step 缓慢上涨，但常规总量指标无法告诉你哪条代码路径留下了对象。

## 1. Memory Snapshot 泄露归因系统图

![系统地图：L03 Memory Snapshot 泄露归因：Memory Snapshot 泄露归因系统图](outputs/system-map-01.png)

系统图从长期 OOM 进入：总量指标只能说明显存变高，snapshot 才能把 allocate/free 事件、live allocation 和 caller stack 串成可定位的泄露候选。

## 2. 事件、live set、caller stack 概念图

![系统地图：L03 Memory Snapshot 泄露归因：事件、live set、caller stack 概念图](outputs/concept-map-01.png)

这节课的概念依赖是 events 先形成时间线，live set 表示还没释放的块，caller stack 把块归因到调用路径，stack 聚合再把一堆单点事件变成排查优先级。

## 3. 本课边界

- CPU mock 只验证 snapshot 数据模型和归因逻辑。
- 真实 CUDA allocator 还要处理异步、stream、fragmentation 和 PyTorch 内部缓存。
- 排查结论要保留原始事件、聚合表和最小复现路径。
