# L08.7 · Speculative Decoding：Greedy Verify

> 本关只做一件事：**实现 spec decode 的 verify 步骤**——target 一次 forward 验证 k 个 draft token。

## 闭环

```bash
cat labs/l24_spec_decode/patch/task.md
$EDITOR labs/l24_spec_decode/patch/starter/spec_decode.py
make patch-test M=l24_spec_decode
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_all_drafts_accepted` | 全对 → k+1 token, num_accepted=k |
| `test_first_mismatch_at_position_2` | 部分对 → 接受到第一个错为止 |
| `test_zero_drafts_one_bonus` | k=0 → 只取 target argmax |
| `test_all_mismatch` | 全错 → 1 token (target argmax @ pos 0) |
| `test_output_dtypes` | 返回 List[int] + int |

## 卡住怎么办

1. 看 `notebooks/n18_spec_decode_acceptance.ipynb`。
2. `make patch-hint M=l24_spec_decode`。
3. `make patch-show-solution M=l24_spec_decode`。

## 进入下一关

`make patch-test` 全绿后，继续做源码理解口试。下一关 [L09 SGLang PD](../l26_sglang_pd_observability/README.md)。
