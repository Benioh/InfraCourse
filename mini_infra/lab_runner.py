from __future__ import annotations

import argparse
import json
import subprocess
import sys
from mini_infra.observability.io import ROOT, utc_now, write_json

COMMANDS: dict[str, list[list[str]]] = {
    "l01_env_conda_cuda": [
        ["-m", "mini_infra.observability.evidence", "--root", "runs/mini_infra"]
    ],
    "l02_pytorch_systems": [
        [
            "-m",
            "mini_infra.training.train_tiny",
            "--backend",
            "simulated",
            "--run-id",
            "{run_id}",
        ]
    ],
    "l03_nccl_ddp_smoke": [["-m", "mini_infra.distributed.collectives", "--json"]],
    "l04_gpu_kernel": [["-m", "mini_infra.gpu.microbench", "--run-id", "{run_id}"]],
    "l05_distributed_primitives": [
        [
            "-m",
            "mini_infra.megatron.run_pretrain",
            "--run-id",
            "{run_id}",
            "--tp",
            "2",
            "--pp",
            "2",
        ]
    ],
    "l06_torchtitan_training": [
        ["-m", "mini_infra.megatron.run_pretrain", "--run-id", "{run_id}"]
    ],
    "l08_megatron_text_pretrain": [
        [
            "-m",
            "mini_infra.megatron.run_pretrain",
            "--run-id",
            "{run_id}",
            "--tp",
            "2",
            "--pp",
            "2",
            "--distributed-optimizer",
        ]
    ],
    "l09_long_context_cp": [["-m", "mini_infra.megatron.run_pretrain", "--run-id", "{run_id}", "--cp", "2", "--seq", "16384", "--yarn-scale", "1.0"]],
    "l11_megatron_scale_optimization": [
        [
            "-m",
            "mini_infra.megatron.run_pretrain",
            "--run-id",
            "{run_id}",
            "--tp",
            "2",
            "--pp",
            "2",
            "--distributed-optimizer",
        ]
    ],
    "l13_moe_ep": [["-m", "mini_infra.megatron.run_pretrain", "--run-id", "{run_id}", "--moe-experts", "8", "--moe-topk", "2", "--ep", "2"]],
    "l17_megatron_multimodal_data": [
        [
            "-m",
            "mini_infra.data.manifest",
            "--output",
            "runs/mini_infra/data/{run_id}_multimodal_manifest.json",
        ]
    ],
    "l18_data_engineering": [["-m", "mini_infra.data.wds_pipeline", "--run-id", "{run_id}"]],
    "l19_vllm_serving_baseline": [["-m", "mini_infra.vllm.run_engine"]],
    "l21_sglang_serving_core": [["-m", "mini_infra.sglang.run_scheduler"]],
    "l23_quant_serving": [["-m", "mini_infra.vllm.run_engine", "--run-id", "{run_id}", "--quant", "awq"]],
    "l24_spec_decode": [["-m", "mini_infra.vllm.run_engine", "--run-id", "{run_id}", "--spec", "ngram"]],
    "l26_sglang_pd_observability": [["-m", "mini_infra.sglang.run_scheduler", "--pd"]],
    "l29_verl_rl_baseline": [["-m", "mini_infra.rl.reward"]],
    "l31_rollout_only_smoke": [["-m", "mini_infra.slime.ray.rollout"]],
    "l32_slime_rl_core": [["-m", "mini_infra.slime.train", "--steps", "2"]],
    "l35_multimodal_capstone": [["-m", "mini_infra.reports.build_delivery"]],
}


def _render(command: list[str], run_id: str) -> list[str]:
    return [part.format(run_id=run_id) for part in command]


def run_mission(mission: str, run_id: str) -> dict:
    if mission not in COMMANDS:
        known = ", ".join(sorted(COMMANDS))
        raise SystemExit(f"Unknown mission {mission!r}. Known missions: {known}")
    results = []
    for command in COMMANDS[mission]:
        rendered = _render(command, run_id)
        completed = subprocess.run(
            [sys.executable, *rendered],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(
            {
                "command": "python " + " ".join(rendered),
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )
    output = ROOT / "runs" / "mini_infra" / "lab_bridge" / mission / f"{run_id}.json"
    payload = {
        "mission": mission,
        "run_id": run_id,
        "created_at": utc_now(),
        "results": results,
        "note": "MiniInfra lab bridge output; use lab smoke artifacts for real performance claims.",
    }
    write_json(output, payload)
    return {
        "status": "ok",
        "output": str(output.relative_to(ROOT)),
        "commands": len(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the MiniInfra bridge for one InfraCourse mission"
    )
    parser.add_argument("--mission", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_id = args.run_id or args.mission
    print(json.dumps(run_mission(args.mission, run_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
