# L30 Source Reading Card

## 主路径

| 顺序 | 文件 | 读什么 |
|---|---|---|
| 1 | `patch/starter/multiturn_tokenizer.py` | base、delta、role mask 的 TODO |
| 2 | `patch/reference/multiturn_tokenizer.py` | Fixed Base 的完整最小实现 |
| 3 | `patch/_mock_tokenizer.py` | default system injection 与 QwQ think drop |
| 4 | `patch/tests/test_patch.py` | 8 个行为边界 |

## 读源码时的问题

1. base 为什么不能随真实 messages 改变？
2. 每条 msg 为什么要单独渲染为 `BASE + [msg]`？
3. assistant 和 tool 的 loss mask 有什么差异？
4. QwQ think drop 在 mock tokenizer 哪个分支发生？
5. 默认 system 泄漏会在哪个测试里暴露？

## 记忆点

- 固定 base 解决模板前缀稳定性。
- Delta 解决 message 到 token 的边界定位。
- Role mask 决定哪些 token 是训练目标。
- Mock tokenizer 是验证条件渲染的工具，不替代真实 tokenizer 端到端检查。
