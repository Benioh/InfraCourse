# L08 Megatron 数据预处理复盘模板

## Run 信息

- 日期：
- 机器 / GPU：
- 命令：
- 配置文件：
- git commit：
- Python / NumPy / tokenizer 版本：
- 输入数据路径：
- 输出 prefix：

## 输入数据

- JSONL 行数：
- 有效 text 样本数：
- 空 text / 缺失字段数量：
- tokenizer：
- special token / append_eod 设置：
- dtype：

## 输出文件

| artifact | 路径 | 检查结果 |
|---|---|---|
| `.bin` |  |  |
| `.idx` |  |  |
| `preprocess_summary.json` |  |  |
| `sample_dump.json` |  |  |
| `real_megatron_command.sh` |  |  |

## 指标

| 指标 | 数值 | 判断 |
|---|---:|---|
| n_samples |  |  |
| total_tokens |  |  |
| bin bytes |  |  |
| dtype bytes |  |  |
| accepted |  |  |

## Roundtrip 抽查

| sample index | 原文本摘要 | token len | head tokens | 是否匹配 tokenizer |
|---:|---|---:|---|---|
| 0 |  |  |  |  |
| 1 |  |  |  |  |
| last |  |  |  |  |

## 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
|  |  |  |

## 后续训练连接

- Megatron `--data-path`：
- train / valid / test split：
- resume 时需要固定的数据相关参数：

## 结论

- 本次能证明什么：
- 不能证明什么：
- 下一步要改的数据、tokenizer 或训练配置：
