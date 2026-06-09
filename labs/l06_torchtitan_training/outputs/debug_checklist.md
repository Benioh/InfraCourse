# L07 Activation Checkpoint Debug Checklist

## 1. 先固定现场

- 命令：patch-test、notebook、stub 还是真实 TorchTitan。
- 环境：CPU/GPU、PyTorch 版本、CUDA 是否可用。
- policy：包哪些 child，未包哪些 child。
- 指标：output diff、input grad diff、param grad diff、wrapped count、peak memory、step time。

## 2. 输出不等价

- wrapper 是否调用 `checkpoint(self.module, *args, use_reentrant=False, **kwargs)`。
- wrapped 和 base 是否来自同一初始权重。
- 模型里是否有 dropout 或随机逻辑。
- policy 是否误包了改变外部语义的模块。

对应源码：

- `patch/reference/selective_ckpt.py` L13-L19
- `patch/tests/test_patch.py` L65-L75

## 3. 梯度不等价

- 是否复制了参数或新建了同结构模块。
- 原 child 是否被保存到 wrapper 的 `self.module`。
- 参数仍是否在 module tree 中。
- 参数名变化是否被误判成参数值变化。

对应源码：

- `patch/reference/selective_ckpt.py` L22-L26
- `patch/tests/test_patch.py` L78-L99

## 4. Policy 计数错误

- 是否使用 `name.lower()`。
- 是否同时匹配 `attn` 和 `attention`。
- 是否只遍历 immediate children。
- 全 False policy 是否没有替换任何直接 child。

对应源码：

- `patch/reference/selective_ckpt.py` L29-L31
- `patch/tests/test_patch.py` L102-L123

## 5. GPU Memory 结果

| 状态 | 解释 | 报告写法 |
|---|---|---|
| pass | CUDA 上 peak memory 下降 | 写明模型、shape、base_peak、wrap_peak |
| skip | 没有 CUDA | 写成未验证 |
| failure | peak 没降或测试异常 | 同时记录 peak、shape、dtype、PyTorch 版本 |

显存结论还要配 step time。checkpoint 省显存通常会增加计算。

## 6. TorchTitan 迁移边界

- activation checkpoint 应与 compile/FSDP 顺序一起看。
- full/selective/memory-budget 模式语义不同。
- 训练 checkpoint save/load 与 activation checkpoint 分开排查。
- 真实 TorchTitan 集群训练需要用官方训练入口和配置验证，stub 只证明课程 artifact 链路。
