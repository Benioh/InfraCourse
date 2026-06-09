# Debug Checklist：L37 CUDA IPC Weight Sync

## 1. 固定现场

- 记录命令、git commit、Python/PyTorch 版本、GPU/CPU 模式、world_size、tp_rank 和输入 tensor shape/dtype。
- 保存 patch-test 输出、handle bytes、tensor bytes、data_ptr 对比、rank gather 结果和 pool size。
- 标记运行类型：CPU patch、notebook、真 CUDA IPC multiprocessing、SGLang endpoint 或 SLiME co-locate。

## 2. 先查 handle 合同

- `serialize_handle` 的 blob 是否远小于 tensor data。
- blob 是否只包含 handle、shape、dtype、stride、device 等 meta。
- `deserialize_handle` 是否返回共享 storage，而不是 clone。
- 源 tensor 生命周期是否覆盖接收端使用窗口。

## 3. 再查 rank 和 LST

| 阶段 | 要看什么 | 常见动作 |
|---|---|---|
| gather | 所有 rank 是否写入，非 0 rank 是否返回 None | 查 collective 参与 rank 和 world_size |
| rank 0 | 是否等到所有 rank 后再返回完整列表 | 避免返回部分列表 |
| LST | `values[rank]` 是否对应当前 TP rank | 检查 values 顺序和 TP rank 映射 |
| unwrap | SGLang 是否先 unwrap 再 load weights | 查 ModelRunner update 日志 |
| flush | 是否只在一批参数更新完成后清 cache | 查 flush_cache、pool size 和 version |

## 4. 生产路径分层

- SLiME 是否 pause generation 后再发权重。
- bucket metadata 是否包含 name、shape、dtype 和 offset。
- `MultiprocessingSerializer` 是否走 tensor reduction，而不是普通数据 pickle。
- `torch.cuda.ipc_collect()` 是否在消费者关闭 IPC handle 后执行。
- `weight_version` 是否随 update 推进。

## 5. 结束条件

- 能用最小命令复现问题。
- 能指出问题位于 handle 序列化、rank gather、LST unwrap、bucket、loader、flush 或生命周期。
- 结论写入 `rl_rollout_template.md`，并列出下一次只改一个变量的实验。
