# L19：Data Engineering - MinHash, WebDataset, Shard Resume

L19 处理训练前的数据工程问题。数据量变大后，质量问题和供给问题会同时出现：训练集里有近重复文本，eval 可能被训练数据污染，shard 太碎会拖慢 DataLoader，坏 shard 或重启会造成重复读或漏读。本讲先用 MinHash 解决文本近重复，再把结果放回 WebDataset-style pipeline 和 shard resume 的系统边界里。

## 0. 学完后要能做什么

- 解释 shingle、Jaccard、MinHash signature 和 LSH banding 的关系。
- 实现 `minhash_signature`、`jaccard_estimate`、`dedup` 的最小合同。
- 说明 `num_perm` 对估计误差、计算量和存储量的影响。
- 用 first-occurrence 策略保留样本索引，并维护 manifest 行号可追踪。
- 读懂 MiniInfra 中 WDS throughput、detshuffle、corrupt shard recovery 的模拟路径。

## 1. 问题背景：重复数据影响 token 预算和评测可信度

预训练数据经常来自网页镜像、转载文章、模板页面、代码仓库和 caption 数据。完全重复可以用 SHA-256 发现，但轻微改写、换一个词、页眉页脚不同的近重复文本需要集合相似度。

重复样本带来两类风险。第一类是训练预算浪费：同一内容多次出现会占据 token budget。第二类是评测污染：训练集和 eval 集重叠时，指标混入记忆能力，无法判断模型是否学会泛化。多模态 caption 数据还要分清文本去重和图像去重：MinHash 适合 caption 文本；图像本身通常用 pHash、dHash 或 CLIP embedding。

## 2. Shingle 和 Jaccard：先把文本变成集合

shingle 是文本里的局部片段。本 patch 用 char-level n-gram；MiniInfra 示例用 word-level width-gram。二者服务同一个目的：把文本转成一个离散集合，再用 Jaccard 衡量相似度。

Jaccard 的定义是：

```text
J(A, B) = |A ∩ B| / |A ∪ B|
```

shingle 宽度决定误差类型。宽度太小，主题相近的文本容易被判成重复；宽度太大，轻微改写会漏掉。短 caption、多语言和代码数据常用 char-level，因为它不依赖 tokenizer；长文档常用 word-level，因为短语级片段更稳定。

## 3. MinHash：用固定长度签名估计 Jaccard

MinHash 的核心性质：对两个集合使用同一个随机排列时，它们最小元素相同的概率等于 Jaccard。工程上不会真的生成巨大排列，而是用多组稳定哈希函数模拟多次抽样。

`minhash_signature(text, num_perm, n_gram)` 做四步：

1. 生成 shingle 集合。
2. 用稳定哈希把每个 shingle 变成整数。
3. 对每个 permutation seed 生成一组 `(a, b)`。
4. 对所有 shingle 计算 `(a * h + b) mod p`，取最小值作为这一维签名。

签名长度等于 `num_perm`。签名越长，估计方差越小；成本也按 `num_perm` 线性增加。10 亿文档下，64 维和 256 维签名的存储差别会直接影响集群成本。

## 4. Jaccard 估计：比较签名相同位置比例

`jaccard_estimate(sig1, sig2)` 只比较两个签名相同位置的比例：

```text
estimate = matches / len(signature)
```

相同文本的签名应该逐维相同，估计值为 1.0。近重复文本的估计值较高，无关文本较低。测试里用 128 维签名检查“改一个词”的句子估计值大于 0.7，无关句子低于 0.2。

常见错误是使用 Python 内置 `hash()`。它可能受进程 hash seed 影响，不能作为可复现实验的基础。L19 的 reference 用 md5，MiniInfra 用 sha1，目的都是稳定。

## 5. Dedup：first-occurrence 保留索引

`dedup(texts, threshold, num_perm, n_gram)` 先为每条文本生成签名，再从前往后扫描。当前样本只要和任何已保留样本的估计 Jaccard 大于等于阈值，就判为重复；否则保留当前索引。

返回索引而不是返回文本，是为了保留外部样本 id、manifest 行号、caption 文件和训练日志之间的对应关系。first-occurrence 策略简单稳定，但输入顺序会影响保留哪一份。真实数据上必须抽样审计 detected pairs，不能只看 dedup ratio。

## 6. LSH Banding：大规模候选检索

朴素 dedup 是 O(N^2)。LSH banding 把签名切成多个 band，每个 band 包含若干行。两个文档只要在某个 band 上完全相同，就进入候选集合；候选对再计算签名相似度或真实 Jaccard。

检测概率：

```text
P(candidate) = 1 - (1 - J^r)^b
```

其中 `J` 是 Jaccard，`r` 是每个 band 的行数，`b` 是 band 数。`b * r` 必须等于 `num_perm`。banding 不改变相似度定义，只减少候选对数量。

## 7. WebDataset 与 Shard Resume：数据质量之外还有供给问题

MinHash 解决内容重复，WebDataset-style pipeline 解决数据供给。`mini_infra/data/wds_pipeline.py` 模拟了 shard list、detshuffle、workers、prefetch、near duplicate report 和 corrupt shard recovery。它输出 `wds_throughput_mbs`、`p99_batch_latency_ms`、`dedup_ratio`、`recovered_shards`、`lost_samples` 等指标。

`mini_infra/data/shard_resume.py` 处理两件事：坏 shard 进入 skipped 列表，resume cursor 让重启后从指定 shard 后继续。detshuffle 用 seed 和 shard 名生成稳定顺序，保证重跑时顺序可复现。

数据慢时不要直接改模型。先看 shard 大小、磁盘顺序读、worker 数、prefetch、decode 和 IPC。发现重复读或漏读时，先看 shard order、worker state 和 cursor。

## 8. Lab 验收

patch-test 覆盖 5 个边界：

| 测试 | 验收点 |
|---|---|
| `test_signature_size` | 签名长度等于 `num_perm` |
| `test_identical_text_same_signature` | 相同文本签名相同，估计值为 1.0 |
| `test_jaccard_estimate_high_for_near_duplicates` | 近重复高、无关低 |
| `test_dedup_removes_exact_duplicates` | 完全重复被删除，保留 first occurrence |
| `test_dedup_preserves_distinct` | 不相似文本全部保留 |

smoke 用半任务 runner 生成数据工程 artifact：

```bash
python labs/l18_data_engineering/scripts/run_smoke.py --run-id l19_validation
```

验证时分层看：patch-test 证明函数合同，notebook 帮你观察参数影响，smoke artifact 证明数据 pipeline 能产出吞吐、dedup 和 recovery 证据。

## 9. 小结

L19 的主线是”质量 + 供给 + 容灾”。MinHash 把近重复文本变成可估计的集合问题；dedup 把相似度阈值变成保留索引；WebDataset-style pipeline 把过滤后的数据送进训练；shard resume 保证坏数据和重启不会静默破坏样本顺序。

---

## 补充：MinHash 数学直觉与 LSH 参数选择

### 为什么 MinHash 能估计 Jaccard？（概率论直觉）

想象把集合 A∪B 中所有元素随机排列。问：”第一个元素（最小值）属于 A∩B 的概率是多少？”

答案正好是 Jaccard：

```
P(min(A∪B) ∈ A∩B) = |A∩B| / |A∪B| = J(A,B)
```

因为随机排列下，A∪B 中每个元素成为最小值的概率相等（= 1/|A∪B|），而 A∩B 中有 |A∩B| 个元素。

**推论**：如果用同一个哈希函数 h 把所有元素映射到整数，然后取 min(h(x) for x in A) 和 min(h(x) for x in B)，这两个最小值相等的概率就等于 J(A,B)。

多次独立实验（num_perm 个不同的哈希函数）取平均，就得到 Jaccard 的无偏估计。误差标准差约为：

```
σ ≈ sqrt(J(1-J) / num_perm)
```

### num_perm 的选择

| num_perm | 误差 (J=0.8) | 误差 (J=0.5) | 存储 (per doc) | 适用场景 |
|---|---|---|---|---|
| 64 | ±0.05 | ±0.06 | 256 bytes | 粗筛，百万级 |
| 128 | ±0.035 | ±0.044 | 512 bytes | 标准，千万级 |
| 256 | ±0.025 | ±0.031 | 1KB | 精确，亿级 |

10 亿文档 × 256 维 = 1TB 签名存储。这是真实的集群成本。

### LSH Banding 参数选择

给定 num_perm = b × r（b 个 band，每个 r 行），检测概率为：

```
P(candidate | J) = 1 - (1 - J^r)^b
```

目标是选择 b, r 使得：
- 在 J = threshold 处，P ≈ 0.5（阈值附近的过渡带）
- J > threshold 时 P 接近 1（高召回）
- J < threshold - 0.1 时 P 接近 0（低误报）

**常用配置**：

| num_perm | threshold | b | r | P(J=threshold) |
|---|---|---|---|---|
| 128 | 0.8 | 32 | 4 | ~0.47 |
| 128 | 0.5 | 16 | 8 | ~0.46 |
| 256 | 0.8 | 64 | 4 | ~0.47 |

### Eval 集污染检测

训练集去重不够——还要检测训练集和 eval 集之间的重叠：

```python
def detect_contamination(train_signatures, eval_signatures, threshold=0.8):
    “””检测 eval 集中有多少样本和训练集”过于相似”。”””
    contaminated = []
    for i, eval_sig in enumerate(eval_signatures):
        for j, train_sig in enumerate(train_signatures):
            if jaccard_estimate(eval_sig, train_sig) >= threshold:
                contaminated.append((i, j))
                break
    return contaminated
```

被污染的 eval 样本应该从评测中排除，或者在报告中单独标注。否则你的 perplexity / accuracy 混入了记忆能力。

### 多模态数据去重

文本去重用 MinHash。图像/视频去重通常用：
- **pHash / dHash**：感知哈希，对轻微缩放/压缩鲁棒
- **CLIP embedding cosine similarity**：语义级去重
- **文件级 SHA-256**：完全相同的文件

Caption 数据需要**两阶段**去重：先图像去重（去掉相同图片的不同 caption），再文本去重（去掉不同图片但相同 caption 的模板文本）。
