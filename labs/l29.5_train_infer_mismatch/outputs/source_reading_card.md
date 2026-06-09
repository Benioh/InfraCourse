# Source Reading Card：L32 Train-Infer Mismatch

## 主路径

1. `labs/l29.5_train_infer_mismatch/patch/starter/mismatch.py`：六个待实现算子的输入、输出和 TODO。
2. `labs/l29.5_train_infer_mismatch/patch/reference/mismatch.py`：K3、TIS、MIS、Geometric IS、Veto、Batch Normalize 的最小 PyTorch 实现。
3. `labs/l29.5_train_infer_mismatch/patch/tests/test_patch.py`：10 个数值和边界合同。
4. `github_repo/slime/examples/train_infer_mismatch_helper/mis.py`：生产分支中的 masked stats、TIS/RS、veto、batch norm 和 metrics。

## 阅读顺序

1. 先看 reference 的 K3、TIS/MIS 和 Geometric IS，确认公式方向。
2. 再看 tests，把每个断言对应到一个行为合同。
3. 然后读 SLiME `compute_mis_weights` 的输入校验和 log ratio 计算。
4. 最后看 TIS、RS、veto、batch norm 和 final metrics 的写入位置。

## 自检

- 我能否说清 `logp_new - logp_old` 的方向？
- 我能否指出 TIS 与 MIS 在源码里的边界动作差异？
- 我能否解释 Geometric IS 为什么必须屏蔽 padding？
- 我能否列出生产排查至少需要哪些 ratio、mask 和 batch norm 指标？
