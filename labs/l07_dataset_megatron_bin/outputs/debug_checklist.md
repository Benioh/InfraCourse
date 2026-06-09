# L08 Megatron 数据预处理 Debug Checklist

这份 checklist 用于排查 JSONL 到 Megatron `.bin/.idx` 的预处理问题。按顺序查，避免直接改训练参数。

## 1. 先固定输入和配置

- [ ] 输入 JSONL 路径是否正确。
- [ ] 每行是否是合法 JSON。
- [ ] 目标字段是否叫 `text`，真实 Megatron 配置是否使用了正确 `--json-keys`。
- [ ] 空文本、缺失字段和空 tokenizer 输出比例是多少。
- [ ] tokenizer 名称、vocab 文件、special token 配置是否记录。
- [ ] dtype 是 `int32`、`uint16` 还是 `int64`。
- [ ] output prefix 是否指向本次运行目录或预期数据目录。

## 2. 检查输出文件

- [ ] `<prefix>.bin` 是否存在且非空。
- [ ] `<prefix>.idx` 是否存在且能读 header。
- [ ] idx 中的 magic、version、dtype code 是否符合预期。
- [ ] `n_samples` 是否等于有效文本样本数。
- [ ] `total_tokens` 是否等于所有写入样本 token 数之和。
- [ ] `.bin` 字节数是否等于 `total_tokens * dtype_bytes`。
- [ ] offsets 是否单调递增。
- [ ] lengths 是否全部为正。

## 3. 检查 roundtrip

- [ ] 用 `IndexedDataset(prefix)` 打开成功。
- [ ] `len(dataset)` 是否等于 `n_samples`。
- [ ] 抽样 `dataset[0]`、`dataset[1]`、最后一条是否非空。
- [ ] 抽样读回结果是否等于 tokenizer 对原文本的输出。
- [ ] 越界下标是否抛 `IndexError`。

## 4. 常见失败判断

| 现象 | 优先检查 |
|---|---|
| `n_samples` 偏少 | 空 text、json key 错、tokenizer 返回空 |
| `total_tokens` 偏少 | 清洗规则、截断、sentence split、tokenizer 配置 |
| `uint16` 报错 | vocab 或 special token 超过 65535，改用 int32 |
| Megatron 读不到数据 | `--data-path` 是否传 prefix，而不是单独 `.bin` 或 `.idx` |
| 读回 token 错乱 | dtype code、offset 单位、length 单位是否混淆 |
| 恢复训练样本错位 | 数据 split、shuffle seed、document index 和 checkpoint 采样位置 |

## 5. 跑 drill 后必须保存

- [ ] `config.resolved.yaml`
- [ ] `artifacts/preprocess_summary.json`
- [ ] `artifacts/sample_dump.json`
- [ ] `artifacts/real_megatron_command.sh`
- [ ] `metrics.jsonl`
- [ ] `report.md`

## 6. 结束条件

复盘结束时应能写清：

- 输入数据路径和有效样本数。
- tokenizer 和 dtype。
- 输出 prefix。
- `.bin` 字节数、`n_samples`、`total_tokens`。
- 抽样 roundtrip 结果。
- 后续 Megatron `--data-path` 命令。
