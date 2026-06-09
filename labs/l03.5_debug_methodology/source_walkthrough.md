# L03.5 源码带读：Tensor 对齐与最小复现工具

## 1. tensor_alignment_demo.py

展示如何逐层对比两个模型的输出。

### 核心流程

```python
def compare_models(model_a, model_b, input_batch):
    """逐层对比两个模型的中间输出"""
    hooks_a, hooks_b = [], []
    outputs_a, outputs_b = [], []

    # 注册 hook 捕获中间层输出
    for name, module in model_a.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm)):
            hooks_a.append(module.register_forward_hook(
                lambda m, i, o, n=name: outputs_a.append({"name": n, "tensor": o})
            ))

    for name, module in model_b.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm)):
            hooks_b.append(module.register_forward_hook(
                lambda m, i, o, n=name: outputs_b.append({"name": n, "tensor": o})
            ))

    # Forward
    with torch.no_grad():
        model_a(input_batch)
        model_b(input_batch)

    # 逐层对比
    for a, b in zip(outputs_a, outputs_b):
        report = tensor_diff_report(a["tensor"], b["tensor"])
        print(f"{a['name']:30s}  cosine={report['cosine_similarity']:.6f}")
```

### 你应该观察到什么

- 如果两个模型实现相同，所有层的 cosine sim 应该 ≈ 1.0（FP16 下 > 0.9999）。
- 如果某一层突然下降（如从 0.9999 跳到 0.95），那一层就是 bug 所在。
- 误差会在深层累积放大——所以要找**第一个**显著偏离的层。

## 2. 阅读顺序

1. 理解 `tensor_diff_report` 的三个指标（abs、rel、cosine）各自适合什么场景。
2. 理解 `find_first_diverge_layer` 的逻辑：为什么找第一个偏离点很重要。
3. 理解 `make_minimal_repro_config` 的缩小策略：为什么这些维度需要缩小。
4. 理解 `classify_symptom` 的模式匹配：建立"症状→原因"的直觉。
