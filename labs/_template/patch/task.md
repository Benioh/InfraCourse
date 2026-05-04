# LXX Patch · <一句话任务>

## 你要交付什么

在 `starter/<file>.py` 里把 `<Class/Function>` 实现完整，使其满足下列契约。

**禁止使用** `<某些框架快捷 API>`，因为本关的目的就是让你自己写一遍。
**允许使用** `<低层 API 列表>`。

补丁规模目标：N–M 行 Python（不含空行注释）。

## 接口契约

```python
class <ClassName>:
    def __init__(self, ...): ...
    def forward(self, ...): ...  # 或 method 名
```

## 不变量（写代码时心里要装着）

1. **数学正确性**：……
2. **通信正确性**：……
3. **性能边界**：……
4. **边界条件**：……

## 怎么验证

```bash
make patch-test M=lXX_xxx
```

测试覆盖：

| 级别 | 测试名 | 说明 |
|---|---|---|
| 数值 | `test_*_matches_*` | 与 reference 实现 `allclose` |
| 行为 | `test_*_triggers_*` | 特定输入下产生特定行为 |
| 性能 | `test_*_perf_gate` | 必须越过 throughput / memory 门槛 |

## 卡住怎么办

1. 先在 `notebooks/nXX_*.ipynb` 把概念图画一遍。
2. 30 分钟解不出，运行 `make patch-hint M=lXX_xxx` 查看 TODO 列表。
3. 仍然卡住，运行 `make patch-show-solution M=lXX_xxx` 看参考解。

## 写完之后你能做什么

- 解释 `<framework>/...` 的真实实现每一行。
- 在面试里讲清 `<concept>` 的工程取舍。
- 在后续关卡（L??）复用这个 primitive。
