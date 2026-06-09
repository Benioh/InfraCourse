# L17 Source Reading Card：Crash Resume

## 主路径

1. `labs/l16_resume_after_crash/patch/starter/crash_safe.py`：最小 atomic save / load latest / save step 合同。
2. `labs/l16_resume_after_crash/patch/reference/crash_safe.py`：tmp+fsync+replace、tmp cleanup、latest marker。
3. `labs/l16_resume_after_crash/patch/tests/test_patch.py`：6 条 crash-resume 行为不变量。
4. `labs/l16_resume_after_crash/scripts/run_crash_drill.py`：baseline vs crash+resume loss 对比。
5. `mini_infra/megatron/training/checkpointing.py`：教学版 payload 与 latest marker。
6. `github_repo/Megatron-LM/megatron/training/checkpointing.py`：Megatron tracker、save finalization、optimizer 和 RNG load。

## 关键行

| 文件 | 行 | 读完要得到的结论 |
|---|---|---|
| patch starter | L13-L47 | 学生要实现 tmp 写入、load cleanup 和 save_step 顺序 |
| patch reference | L13-L24 | atomic_save 的提交边界是 `os.replace` |
| patch reference | L27-L56 | load_latest 先删除 tmp，再选择 committed checkpoint |
| patch reference | L59-L79 | save_step 先保存 step 文件，再更新 marker |
| patch tests | L21-L109 | 测试覆盖 tmp 清理、状态恢复、幂等保存和 loss 对齐 |
| run_crash_drill | L73-L125 | drill 把 baseline/resume loss 差异落盘 |
| Megatron checkpointing | L770-L835 | tracker 更新在 save finalize 阶段发生 |
| Megatron checkpointing | L1918-L2000 | load 恢复 optimizer、scheduler 和 RNG |

## 先跳过

- 多机 elastic restart：先读通单进程文件提交边界。
- 异步 checkpoint backend 的实现细节：先看 finalize 顺序。
- 分布式文件系统一致性模型：本讲只建立最小文件合同。
- 真实模型非确定性 kernel：CPU drill 使用确定性 mock 更新。

## 自检

- 我能否解释 `.tmp` 和 committed checkpoint 的区别？
- 我能否指出 marker 更新在 patch 和 Megatron 中分别发生在哪里？
- 我能否说明 loss 对齐为什么比文件存在更强？
- 我能否判断一次恢复失败来自文件提交、状态缺失，还是随机性变化？
