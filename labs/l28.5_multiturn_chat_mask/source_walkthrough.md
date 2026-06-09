# L30 源码带读：Multi-turn Chat Mask

## 0. 源码地图

```text
labs/l28.5_multiturn_chat_mask/patch/starter/multiturn_tokenizer.py
labs/l28.5_multiturn_chat_mask/patch/reference/multiturn_tokenizer.py
labs/l28.5_multiturn_chat_mask/patch/_mock_tokenizer.py
labs/l28.5_multiturn_chat_mask/patch/tests/test_patch.py
```

## 1. Patch Starter

文件：`labs/l28.5_multiturn_chat_mask/patch/starter/multiturn_tokenizer.py`

阅读顺序：

- L17-L23：`BASE_CONVERSATION` 的 system/user 固定前缀。
- L26-L35：`tokenize_with_loss_mask` 的输入、输出和算法说明。
- L36-L47：TODO 展示 base 渲染、`BASE + [msg]` 渲染和 delta 截取。
- L49-L57：TODO 展示 role mask、attention mask 和返回值。

结论：starter 的所有复杂度都围绕固定 base 和 role mask。

## 2. Patch Reference

文件：`labs/l28.5_multiturn_chat_mask/patch/reference/multiturn_tokenizer.py`

阅读顺序：

- L7-L9：base 包含 system 和 user。
- L13-L18：函数入口先渲染 base_str。
- L19-L28：循环中每条 msg 独立渲染为 `BASE + [msg]`，再截 delta。
- L29-L36：assistant delta 进入 loss，其它 role 写 `-100`，attention 全 1。

结论：reference 没有维护对话历史；它让每条 message 都在固定上下文中单独测量。

## 3. Mock Tokenizer

文件：`labs/l28.5_multiturn_chat_mask/patch/_mock_tokenizer.py`

阅读顺序：

- L18-L27：mock tokenizer 保存两个行为开关，并提供字符级 encode。
- L32-L43：`apply_chat_template` 复制输入，并在缺 system 时注入默认 system。
- L44-L57：QwQ mode 会从非末尾 assistant 中删除 think 内容。
- L59-L68：按 role 拼接字符串，必要时返回 token id。

结论：mock tokenizer 用可控方式模拟真实模板的两个危险分支。

## 4. Patch Tests

文件：`labs/l28.5_multiturn_chat_mask/patch/tests/test_patch.py`

阅读顺序：

- L32-L45：检查三列表等长、attention 全 1、loss 值域合法。
- L48-L66：检查无 assistant 全 mask，以及 assistant 内容能从 loss 解码。
- L69-L86：多轮两个 assistant 都进入 loss，user 内容不进入 loss。
- L88-L99：默认 system 注入 case 的输入样本。
- 第 100-107 行：断言 assistant reply 进入 loss，`DEFAULT_SYS` 不泄漏。
- 第 109-128 行：QwQ think drop case 要保留 `<think>SECRET</think>`、ANS1 和 ANS2。
- 第 131-145 行：tool 消息被 mask，final assistant 保留。

结论：每个测试都对应一个实现分支，失败时按测试名回到对应分支。

## 自检问题

1. 为什么 base 同时需要 system 和 user？
2. `delta_str` 从哪里截出来？
3. 哪些 role 进入 loss，哪些 role 写 `-100`？
4. default system injection 会破坏哪类朴素算法？
5. QwQ think drop 为什么会影响训练 mask？
