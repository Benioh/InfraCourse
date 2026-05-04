#!/usr/bin/env bash
set -euo pipefail

# 使用前请先记录硬件模式：0GPU / 4090 / H200，并确认 envs/base.yaml 已创建。
make check-env

# 基础证据链
make smoke M=l00_env_conda_cuda
make smoke M=l01_pytorch_systems
make smoke M=l01_5_nccl_ddp_smoke
make smoke M=l02_distributed_primitives

# 训练/数据主线：无 GPU 或缺框架时必须在报告中标注 validation-only。
make smoke M=l03_torchtitan_training
make smoke M=l04_megatron_text_pretrain
make smoke M=l05_megatron_scale_optimization
make smoke M=l06_megatron_multimodal_data

# Serving 主线
make smoke M=l07_vllm_serving_baseline
make smoke M=l08_sglang_serving_core
make smoke M=l09_sglang_pd_observability

# RL 主线
make smoke M=l10_verl_rl_baseline
make smoke M=l10_5_rollout_only_smoke
make smoke M=l11_slime_rl_core

# Capstone 汇总
make smoke M=l12_multimodal_capstone
make grade M=l12_multimodal_capstone
make self-check M=l12_multimodal_capstone
