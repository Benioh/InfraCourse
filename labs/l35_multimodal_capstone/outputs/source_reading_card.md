# L41 Source Reading Card

## 主路径

1. `patch/reference/mm_omni.py`：projector、WER、CLIP score。
2. `patch/tests/test_patch.py`：7 个 Bronze 行为测试。
3. `scripts/run_capstone_aggregator.py`：扫描 runs 并生成 final artifacts。
4. `scripts/run_capstone_stub.py`：stub 转调 aggregator。
5. `mini_infra/reports/build_delivery.py`：最小 delivery helper。

## 阅读方法

1. 先看 patch 的输入输出 shape。
2. 再看 WER 和 CLIP score 的数值边界。
3. 继续看 aggregator 如何读取 run 证据。
4. 最后检查 risk register 如何表达边界。
