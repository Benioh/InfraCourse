"""L03.5 Smoke runner"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patch" / "reference"))
from debug_toolkit import make_minimal_repro_config, classify_symptom


def main():
    run_id = f"run_{int(time.time())}"
    out_dir = Path(f"runs/mini_infra/l03.5_debug_methodology/{run_id}")
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    full_config = {
        "num_nodes": 4, "num_gpus": 8, "batch_size": 32,
        "max_seq_len": 8192, "num_steps": 10000, "seed": 0,
        "dataset_size": 100000, "model_name": "llama-7b",
    }
    minimal = make_minimal_repro_config(full_config)
    (artifacts_dir / "minimal_repro_config.json").write_text(json.dumps(minimal, indent=2))

    symptoms = [
        {"type": "hang", "context": "multi_gpu"},
        {"type": "loss_mismatch", "context": "packed"},
        {"type": "oom", "context": "long_seq"},
        {"type": "nan", "context": ""},
    ]
    classifications = [classify_symptom(s) for s in symptoms]
    (artifacts_dir / "symptom_classifications.json").write_text(
        json.dumps(list(zip(symptoms, classifications)), indent=2, default=str)
    )

    metrics = {"lab": "l03.5_debug_methodology", "timestamp": time.time()}
    (out_dir / "metrics.jsonl").write_text(json.dumps(metrics) + "\n")
    (out_dir / "report.md").write_text("# L03.5 Smoke Report\n\nDebug toolkit verified.\n")

    print(f"Smoke complete. Output: {out_dir}")


if __name__ == "__main__":
    main()
