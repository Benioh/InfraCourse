# L03.5 Patch · Debug 工具包

> 实现 4 个函数，建立系统化的 debug 能力。

## 你要改的文件

`labs/l03.5_debug_methodology/patch/starter/debug_toolkit.py`

## 四个要实现的函数

### 1. `tensor_diff_report(a, b) -> dict`

计算两个 tensor 的详细差异报告。

**返回 dict**：
- `max_abs_diff`: `float` — 最大绝对差
- `mean_abs_diff`: `float` — 平均绝对差
- `max_rel_diff`: `float` — 最大相对差（`abs_diff / (b.abs() + 1e-8)`）
- `mean_rel_diff`: `float` — 平均相对差
- `cosine_similarity`: `float` — flatten 后做 cosine similarity

### 2. `find_first_diverge_layer(baseline_outputs, test_outputs, threshold=0.999) -> dict`

逐层对比两个模型的输出，找到第一个 cosine_similarity < threshold 的层。

**输入**：
- `baseline_outputs`: `list[dict]`，每个 dict 包含 `name: str` 和 `tensor: torch.Tensor`
- `test_outputs`: 同上
- `threshold`: cosine similarity 低于此值认为 diverge

**返回 dict**：
- `diverge_layer_idx`: `int` — 第一个 diverge 的层 index（-1 如果全部 aligned）
- `diverge_layer_name`: `str | None` — diverge 层的 name
- `cosine_sim`: `float` — diverge 层的 cosine similarity（全 aligned 时为 1.0）
- `all_aligned`: `bool` — 所有层是否都 aligned

### 3. `make_minimal_repro_config(full_config: dict) -> dict`

从完整训练配置生成最小复现配置。

**缩小规则**：
- `num_nodes` → 1
- `num_gpus` → min(原值, 2)，如果原值 ≤ 1 则保持 1
- `batch_size` → 1
- `max_seq_len` → min(原值, 512)
- `num_steps` → 5
- `seed` → 42
- `dataset_size` → min(原值, 10)
- 其他字段保留不变

### 4. `classify_symptom(symptom: dict) -> dict`

根据症状描述判断问题类别和排查方向。

**输入 `symptom`**：
- `type`: `str` — 五选一：`"loss_mismatch"`, `"hang"`, `"slow"`, `"oom"`, `"nan"`
- `context`: `str` — 额外上下文：`"multi_gpu"`, `"packed"`, `"long_seq"`, `""`

**返回 dict**：
- `category`: `str` — 问题大类
- `first_check`: `str` — 第一步应该检查什么
- `likely_cause`: `str` — 最可能的原因

**分类逻辑**：
- `"hang"` → category: `"distributed_sync"`，first_check 包含 "NCCL"
- `"loss_mismatch"` + context含"packed" → category: `"data_alignment"`，first_check 包含 "attention_mask"
- `"loss_mismatch"` 其他 → category: `"data_alignment"`
- `"slow"` + "multi_gpu" → category: `"performance"`，likely_cause 包含 "communication"
- `"slow"` 其他 → category: `"performance"`
- `"oom"` → category: `"memory"`
- `"nan"` → category: `"numerical_stability"`

## 测试覆盖

| 测试 | 通过条件 |
|---|---|
| `test_tensor_diff_report` | 相同 tensor cosine=1.0；不同 tensor 有正确差异 |
| `test_layer_alignment` | 完全匹配时 all_aligned=True；diverge 时正确定位 |
| `test_minimal_repro_config` | 缩小规则正确，其他字段保留 |
| `test_classify_symptom` | 各类型分类正确 |
| `test_diverge_detection` | slow+multi_gpu 正确分类 |

```bash
make patch-test M=l03.5_debug_methodology
```
