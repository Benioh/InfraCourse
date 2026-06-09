# L30 Debug Tickets

| Ticket | Stage | Symptom | First Check |
|---|---|---|---|
| `multiturn_default_system_leak` | Template | `DEFAULT_SYS` 出现在输出 token 中 | 检查 base 是否包含 system |
| `multiturn_think_dropped` | Template | assistant `<think>` 内容不在 loss 中 | 检查是否用 `BASE + [msg]` 单独渲染 |
| `multiturn_tool_loss_leak` | Role mask | tool observation 出现在 decoded loss | 检查非 assistant 分支是否全部写 `-100` |
| `multiturn_user_loss_leak` | Role mask | user prompt 出现在 decoded loss | 检查 role 判断和 loss_mask extend |
| `multiturn_delta_empty` | Delta | 某些 message 没有 token | 检查 `full_str` 是否以 `base_str` 为稳定前缀 |
