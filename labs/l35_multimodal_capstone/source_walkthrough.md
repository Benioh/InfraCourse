# L41 源码带读：MM-Tiny-Omni Capstone

## 0. 源码地图

```text
labs/l35_multimodal_capstone/patch/starter/mm_omni.py
labs/l35_multimodal_capstone/patch/reference/mm_omni.py
labs/l35_multimodal_capstone/patch/tests/test_patch.py
labs/l35_multimodal_capstone/scripts/run_capstone_aggregator.py
labs/l35_multimodal_capstone/scripts/run_capstone_stub.py
mini_infra/reports/build_delivery.py
```

## 1. Patch Starter

文件：`labs/l35_multimodal_capstone/patch/starter/mm_omni.py`

阅读顺序：

- L22-L34：`MultimodalProjector` 的构造参数。
- L36-L41：image/audio projection 和 image positional embedding 的 TODO。
- L43-L50：forward 输入和返回 dict 约定。
- L52-L61：image/audio 两条路径的 TODO。
- L64-L83：WER 的 word-level edit distance 合同。
- L86-L95：CLIP score 的 cosine 合同。

结论：Bronze patch 锁定 shape、WER 和 cosine 三类底层合同。

## 2. Patch Reference

文件：`labs/l35_multimodal_capstone/patch/reference/mm_omni.py`

阅读顺序：

- L20-L23：image/audio projector 和 image_pos_emb 初始化。
- L25-L37：forward 分别处理缺失模态和双模态。
- L40-L50：WER 初始化 DP 表。
- L51-L57：WER 主循环和归一化返回。
- L60-L63：CLIP embedding 归一化后做内积。

结论：reference 的实现都可以用 CPU 测试验证，不依赖真实 encoder。

## 3. Patch Tests

文件：`labs/l35_multimodal_capstone/patch/tests/test_patch.py`

阅读顺序：

- L26-L32：image-only projector shape。
- L35-L41：audio-only projector shape。
- L44-L51：双模态输出同时存在。
- L57-L69：WER 完全匹配和已知编辑距离。
- L75-L79：相同 embedding 的 CLIP score。
- L82-L89：正交 embedding 的 CLIP score。

结论：测试覆盖 Bronze 数值合同，不覆盖 Stage A/B/C 的真实训练和服务。

## 4. Capstone Aggregator

文件：`labs/l35_multimodal_capstone/scripts/run_capstone_aggregator.py`

阅读顺序：

- L40-L50：读取 YAML/JSON，缺失或解析失败时返回空结果。
- L53-L64：读取 metrics JSONL，并把解析失败行显式保存。
- L67-L79：列出 quests，并查找每个 mission 的最新 run。
- L114-L129：没有 run 时写缺失证据行。
- L131-L152：读取 command、config、metrics、grade、report 和 artifacts。
- L188-L202：把 evidence rows 渲染成 Markdown 表格。
- L264-L298：聚合 debug ticket，写 debug report。
- L301-L327：按 stage 写 stage reports。
- L330-L360：写 risk register。
- L363-L375：写 architecture.mmd，把阶段证据连到 L41 交付包。
- L378-L392：准备 Capstone run 目录和 resolved config。
- L400-L427：写 final artifacts，并记录 metrics。
- L428-L462：写 train.log 和 report.md。

结论：聚合器整理已有证据，不替代真实运行。

## 5. Stub 和 Delivery Helper

文件：`labs/l35_multimodal_capstone/scripts/run_capstone_stub.py`

- L6-L10：stub 直接转调 aggregator。

文件：`mini_infra/reports/build_delivery.py`

- L11-L18：required mission 列表使用当前课程 mission id。
- L21-L33：扫描 evidence 并写 evidence_index。
- L35-L45：写 final_readme 的 required mission 状态。
- L46-L53：写 risk_register。
- L56-L67：CLI 入口。

结论：Capstone 交付包最终要能回到可复查的 run 证据。

## 自检问题

1. Projector 的图像和音频输出 shape 是什么？
2. WER 为什么除以 reference word 数？
3. CLIP score 为什么要先归一化？
4. 聚合器生成哪些 final artifacts？
5. validation-only 行为什么不能支撑真实性能结论？
