# L17 源码带读：Crash Resume

这份带读只走 crash-safe checkpoint 主路径。第一次读时先跳过多机容错框架、异步 checkpoint backend 和具体模型训练细节。目标是看清：临时文件如何提交，启动时如何清理半写文件，step/model/optimizer/RNG 如何恢复，以及 drill 怎样把恢复轨迹变成证据。

## 1. 源码地图

```text
labs/l16_resume_after_crash/patch/starter/crash_safe.py
  -> 学生要补齐的 atomic save / load latest / save step

labs/l16_resume_after_crash/patch/reference/crash_safe.py
  -> tmp+fsync+replace、tmp cleanup、latest marker 和 payload

labs/l16_resume_after_crash/patch/tests/test_patch.py
  -> L17 patch 的行为合同

labs/l16_resume_after_crash/scripts/run_crash_drill.py
  -> baseline vs crash+resume loss 对比

mini_infra/megatron/training/checkpointing.py
  -> 教学版 checkpoint payload 与 marker

github_repo/Megatron-LM/megatron/training/checkpointing.py
  -> Megatron save finalization、tracker、load iteration、optimizer 和 RNG
```

## 2. 阅读步骤一：先看 patch starter 的合同

文件：`labs/l16_resume_after_crash/patch/starter/crash_safe.py`

重点：

- L13-L18：`atomic_save` 必须写 tmp、flush/fsync，再 replace 到正式路径。
- L21-L29：`load_latest` 必须清理 dangling tmp，再返回最新 committed payload 或 `None`。
- L32-L40：`save_step` 的输入包含 step、model、optimizer、RNG 和 extra。
- L41-L47：`save_step` 要构造 payload、保存 step 文件、更新 latest marker 并返回路径。

读完要能回答：进程在 L15、L16、L17 之间任意位置崩溃时，load 应该看到什么？

## 3. 阅读步骤二：reference 的 atomic save

文件：`labs/l16_resume_after_crash/patch/reference/crash_safe.py`

重点：

- L13-L16：目标路径会被规范化，父目录会被创建，tmp 路径由目标 suffix 加 `.tmp` 得到。
- L17-L19：payload 先写入 tmp，并 flush 到文件对象。
- L20-L23：尽量调用 `os.fsync`，不支持时允许跳过。
- L24：`os.replace` 将 tmp 原子提交为正式文件。

读完要能说明：为什么正式文件出现前，tmp 不能被 loader 当成 checkpoint。

## 4. 阅读步骤三：reference 的 load 与 cleanup

文件：`labs/l16_resume_after_crash/patch/reference/crash_safe.py`

重点：

- L27-L35：`_cleanup_partial` 删除目录下所有 `.tmp` 残留。
- L38-L45：`load_latest` 清理后，如果没有目录或没有 committed checkpoint，返回 `None`。
- L46-L54：marker 存在时优先读取 marker 指向的 step 文件。
- L55-L56：marker 不可用时，回退到最高编号 committed checkpoint。

读完要能判断：如果只有 `iter_0000003.pt.tmp`，应该返回 step 3、step 2，还是 `None`？

## 5. 阅读步骤四：reference 的 save_step

文件：`labs/l16_resume_after_crash/patch/reference/crash_safe.py`

重点：

- L59-L69：创建 checkpoint 目录，并用 step 构造固定正式文件名。
- L70-L76：payload 保存 step、model、optimizer、RNG 和 extra。
- L77-L79：先 atomic 保存 step 文件，再 atomic 更新 latest marker。

读完要能解释：为什么 latest marker 更新必须在 checkpoint 文件提交之后。

## 6. 阅读步骤五：用测试反推不变量

文件：`labs/l16_resume_after_crash/patch/tests/test_patch.py`

重点：

- L21-L28：atomic save 后正式文件存在，tmp 消失，JSON 可读。
- L31-L40：只有 tmp 的 step 3 被删除，load 回到 step 2。
- L43-L57：step、optimizer、RNG 和 extra 都要能恢复。
- L60-L71：同 step 重复 save 不增加正式 checkpoint 文件；空目录返回 `None`。
- L74-L88：baseline 连续训练得到完整 loss 序列。
- L89-L100：crash path 在 step 4 保存 model 和 optimizer。
- L101-L109：load 后继续训练，loss 必须和 baseline 对齐。

读完要能说明：哪条测试最能证明 optimizer state 已恢复？

## 7. 阅读步骤六：看 crash drill 的证据链

文件：`labs/l16_resume_after_crash/scripts/run_crash_drill.py`

重点：

- L63-L71：选择实现、读取配置、准备 run 目录和 resolved config。
- L73-L81：用固定 seed 生成 gradients，baseline 连续训练。
- L80-L89：crash path 训练到 crash step，并保存 checkpoint。
- L90-L96：load 后重建状态，继续训练剩余 steps。
- L98-L102：逐步计算 baseline 和 resumed 的相对误差。
- L104-L115：每个 step 的 loss 和 rel diff 写入 `metrics.jsonl`。
- L117-L125：`crash_drill.json` 写入 max_rel_diff、accept、crash_at 和 total_steps。

读完要能说明：`accept=true` 证明的是文件可读，还是恢复轨迹满足当前阈值？

## 8. 阅读步骤七：MiniInfra checkpointing 对照

文件：`mini_infra/megatron/training/checkpointing.py`

MiniInfra 是 L16 的完整 checkpoint 合同，对 L17 的作用是对照 payload 和 marker。

重点：

- L17-L24：save 接收 iteration、model、optimizer、scheduler 和 parallel state。
- L28-L36：payload 中保存 format、iteration、created_at 和各类 state。
- L37-L43：写 checkpoint 文件和 latest marker。
- L46-L57：load 先读 latest marker，并确认 checkpoint 文件存在。
- L60-L68：load 读取 payload 并校验 format。
- L70-L85：比较 parallel_state，返回 warnings 或抛错。

读完要能说明：L16 的 checkpoint 合同和 L17 的 crash-safe 写入合同分别覆盖哪一层风险。

## 9. 阅读步骤八：Megatron save finalization

文件：`github_repo/Megatron-LM/megatron/training/checkpointing.py`

重点：

- L548-L552：保存前收集 RNG state。
- L598-L619：调用 `generate_state_dict` 收集 model、optimizer、scheduler、RNG 和 rerun state。
- L770-L774：checkpoint 保存后准备更新 latest tracker。
- L783-L792：非 local checkpoint 的 finalize callback 写入 release 或 iteration。
- L830-L835：async save 将 marker 更新放进 finalize，sync save 直接执行。

读完要能说明：为什么异步保存不能在任务真正完成前更新 tracker。

## 10. 阅读步骤九：Megatron load 恢复状态

文件：`github_repo/Megatron-LM/megatron/training/checkpointing.py`

重点：

- L1856-L1877：load 后设置 checkpoint version，并从 state_dict 读取 iteration。
- L1879-L1889：恢复 checkpoint args 中的 consumed samples。
- L1903-L1912：把 model state 加载到单个或多个 model chunk。
- L1918-L1953：加载 optimizer 和 scheduler。
- L1974-L2000：加载 RNG state，并设置 Python、NumPy、Torch 和 CUDA RNG。
- L2001-L2026：恢复 tensor parallel RNG tracker，缺失时给出错误路径。

读完要能说明：真实 Megatron 比 L17 patch 多恢复了哪些随机和数据进度状态。

## 11. 可以先跳过的分支

- 异步 checkpoint 的具体 backend：本讲只看 finalize 顺序。
- 分布式文件系统的目录 fsync 与一致性模型：先掌握单文件 committed 边界。
- 多 rank barrier 和 elastic restart：等单进程 crash-safe 写入读通后再扩展。
- 真实模型 kernel 非确定性：本讲 drill 使用可控 mock update。

## 12. 自检问题

1. loader 看到 `.tmp` 时为什么应该删除它？
2. latest marker 先于 checkpoint 文件提交会造成什么风险？
3. 只恢复 model_state，baseline/resume loss 可能从哪里开始分叉？
4. Megatron 在哪些行恢复 optimizer、scheduler 和 RNG？
5. `max_rel_diff` 超过阈值时，你会先看 checkpoint 文件、marker，还是训练数据 seed？
