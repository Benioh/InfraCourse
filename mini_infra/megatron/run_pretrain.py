from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini_infra.megatron.core.context_parallel.ring_attention import ring_attention_plan, seqlen_sweep
from mini_infra.megatron.core.context_parallel.rope import rope_summary
from mini_infra.megatron.core.context_parallel.yarn import yarn_summary
from mini_infra.megatron.core.transformer.moe.experts import moe_summary
from mini_infra.megatron.pretrain_gpt import forward_step, model_provider
from mini_infra.megatron.training.arguments import MiniMegatronArgs
from mini_infra.megatron.training.training import pretrain
from mini_infra.observability.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMegatron pretrain lifecycle")
    parser.add_argument("--run-id")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--seq", type=int, default=32)
    parser.add_argument("--yarn-scale", type=float, default=1.0)
    parser.add_argument("--moe-experts", type=int, default=0)
    parser.add_argument("--moe-topk", type=int, default=2)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--sequence-parallel", action="store_true")
    parser.add_argument("--distributed-optimizer", action="store_true")
    args = parser.parse_args()
    cfg = MiniMegatronArgs(
        seq_length=args.seq,
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        sequence_parallel=args.sequence_parallel,
        use_distributed_optimizer=args.distributed_optimizer,
        save=f"runs/mini_infra/megatron/checkpoints/{args.run_id or 'latest'}",
    )
    result = pretrain(cfg, model_provider, forward_step, run_id=args.run_id)
    run_dir = Path(result["run_dir"])
    if args.cp > 1 or args.seq >= 4096 or args.yarn_scale != 1.0:
        cp_payload = {
            "ring_attention": ring_attention_plan(seq_len=args.seq, cp_size=args.cp),
            "rope": rope_summary(seq_len=args.seq),
            "yarn": yarn_summary(train_seq_len=4096, target_seq_len=max(args.seq, 4096)),
            "seqlen_sweep": seqlen_sweep(),
            "yarn_scale_arg": args.yarn_scale,
        }
        write_json(run_dir / "artifacts" / "seqlen_sweep.json", cp_payload)
        result["context_parallel"] = cp_payload
    if args.moe_experts:
        moe_payload = {
            "dense_baseline": {"active_params": 4096 * 4096 * 2, "alltoall_ms": 0.0},
            "moe": moe_summary(
                num_experts=args.moe_experts, top_k=args.moe_topk, ep_size=args.ep
            ),
            "collapsed_router": moe_summary(
                num_experts=args.moe_experts, top_k=args.moe_topk, ep_size=args.ep, collapse=True
            ),
        }
        write_json(run_dir / "artifacts" / "moe_compare.json", moe_payload)
        result["moe"] = moe_payload
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
