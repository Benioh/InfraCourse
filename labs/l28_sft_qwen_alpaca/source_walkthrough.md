# L29 源码带读：SFT Loss Mask

## 0. 源码地图

```text
labs/l28_sft_qwen_alpaca/patch/starter/sft_pipeline.py
labs/l28_sft_qwen_alpaca/patch/reference/sft_pipeline.py
labs/l28_sft_qwen_alpaca/patch/tests/test_patch.py
labs/l28_sft_qwen_alpaca/scripts/run_sft.py
github_repo/sglang/python/sglang/lang/chat_template.py
github_repo/sglang/python/sglang/srt/utils/hf_transformers/tokenizer.py
```

## 1. Patch Starter

文件：`labs/l28_sft_qwen_alpaca/patch/starter/sft_pipeline.py`

阅读顺序：

- L13-L25：`_validate_messages` 检查空输入、非法 role、连续 user 和结尾 assistant。
- L28-L34：`tokenize_chat_with_loss_mask` 的参数和返回 dict 合同。
- L35-L42：TODO 写出 pad/eos、segment encode、assistant labels、截断和 pad。
- L45-L50：`sft_loss` 只做 ignore-index cross entropy。

结论：starter 把 SFT 的数据合同压缩成两个函数。

## 2. Patch Reference

文件：`labs/l28_sft_qwen_alpaca/patch/reference/sft_pipeline.py`

阅读顺序：

- L13-L25：校验规则和 starter 保持一致。
- L28-L37：`_resolve_pad` 按显式参数、tokenizer pad、tokenizer eos、0 的顺序兜底。
- L40-L50：主函数先校验 messages，再解析 pad/eos。
- L52-L63：assistant segment 保留 labels，并追加 EOS。
- L64-L78：prompt segment 写 `-100`，随后截断、补 pad 和 attention mask。
- L81-L90：`sft_loss` 直接对齐 PyTorch `F.cross_entropy`。

结论：reference 的重点放在每个位置的 label 语义，复杂模板留到下一讲展开。

## 3. Patch Tests

文件：`labs/l28_sft_qwen_alpaca/patch/tests/test_patch.py`

阅读顺序：

- L22-L29：`_CharTokenizer` 提供稳定的字符级 token id、pad 和 EOS。
- L32-L44：输出必须补到 `max_length`，开头不能是 pad。
- L47-L57：user segment 的 labels 全部是 `-100`。
- L60-L72：assistant segment labels 等于 token id，末尾 EOS 也进入 loss。
- L75-L83：pad 位置 attention mask 为 0，label 为 `-100`。
- L86-L94：`sft_loss` 遇到 `-100` 时仍返回有限 loss。

结论：测试覆盖 mask、EOS、pad 和 loss 的最小边界。

## 4. SFT Smoke

文件：`labs/l28_sft_qwen_alpaca/scripts/run_sft.py`

阅读顺序：

- L36-L50：`_impl` 根据 `IMPL` 选择 starter 或 reference，默认 starter 不可用时回退 reference。
- L53-L63：`_synthetic_messages` 构造 system、user、assistant 三段样本。
- L82-L96：CLI、配置、run 目录和 command/config artifact 写出。
- L137-L149：tokenize-only 模式写 metrics 和 report。
- L154-L166：训练模式缺依赖时写 fallback artifact。
- L168-L174：加载 tokenizer、pad token 和模型。
- L201-L207：patch 函数把 messages 转成 tokenized dataset。
- L208-L219：`TrainingArguments` 固定 batch、lr、steps、warmup 和 bf16。
- L220-L227：`Trainer` 接收模型、dataset、tokenizer 和 collator。
- L228-L242：写 trainer metrics 和 `sft_summary.json`。

结论：smoke 的 CPU 路径验证数据合同，train 路径才验证真实框架连接。

## 5. SGLang Chat Template 对照

文件：`github_repo/sglang/python/sglang/lang/chat_template.py`

阅读顺序：

- L22-L33：LLAMA2 风格会根据 role 和历史消息改变 prefix/suffix。
- L43-L54：`get_prompt` 按消息顺序拼接 role prefix、content 和 suffix。
- L61-L70：模板注册和查询是全局 registry。
- L81-L90：default 模板把 system/user/assistant 映射成简单前后缀。

结论：真实模板的前后缀会影响 token 边界，训练和推理必须使用同一套模板语义。

## 6. SGLang Tokenizer Loader 对照

文件：`github_repo/sglang/python/sglang/srt/utils/hf_transformers/tokenizer.py`

阅读顺序：

- L52-L64：先尝试从 tokenizer config 读取声明的 tokenizer class。
- L67-L79：配置缺失、解析失败或没有 class 时返回 None。
- L81-L90：跳过不能直接实例化的 tokenizer 基类，并处理 remote code。
- L137-L149：特殊 tokenizer 名称可能被重定向到本地路径或 GGUF 父目录。
- L163-L172：AutoTokenizer 加载成功后安装 warning filter。

结论：真实 tokenizer 加载路径会受 config、remote code、GGUF 和缓存影响；SFT debug 不能只看训练脚本。

## 自检问题

1. 哪些 token 会进入 labels，哪些 token 会写成 `-100`？
2. EOS 为什么属于 assistant loss？
3. `attention_mask` 和 `labels` mask 的职责有什么区别？
4. CPU smoke 能证明什么，不能证明什么？
5. 真实 tokenizer 的模板注册和加载为什么会影响 SFT 数据？
