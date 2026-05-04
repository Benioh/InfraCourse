# LXX · <一句话本关目标>

> 本关只有一个动作：**改 `patch/starter/<file>.py`，跑过所有测试**。

## 闭环

```bash
cat labs/lXX_xxx/patch/task.md          # 1. 读任务
# 2. 改 patch/starter/<file>.py
make patch-test M=lXX_xxx                # 3. 跑测试，PASS 即过关
```

pytest 全绿代表代码契约通过；框架理解还需要对照 source_reading，并用 AI 框架理解口试检查。

## 你要改的文件

```
labs/lXX_xxx/patch/
├── task.md                       # 任务详细说明（必读）
├── starter/<file>.py             # ★ 唯一要改的文件
├── reference/<file>.py           # 参考解（卡住再看）
└── tests/test_patch.py           # 自动测试（不要改）
```

## 测试覆盖

| 类别 | 测试 | 通过条件 |
|---|---|---|
| 数值 | … | … |
| 行为 | … | … |
| 性能 | … | … |

## 卡住怎么办

1. 看 `notebooks/nXX_*.ipynb` 把概念图画一遍。
2. `make patch-hint M=lXX_xxx` 看 TODO 列表 + 关键提示。
3. `make patch-show-solution M=lXX_xxx` 打开参考解。

## 配套源码研读（可选）

- `github_repo/<framework>/...` —— 看真实工程级实现的边界。
- `mini_infra/<module>/...` —— 看本课同构最小骨架。

## 进入下一关的前置

`make patch-test M=lXX_xxx` 全绿后，再完成本关源码理解口试。
