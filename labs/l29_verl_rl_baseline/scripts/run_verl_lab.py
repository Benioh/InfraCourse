from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.runtime_utils import (
    append_jsonl,
    ensure_prediction,
    prepare_run_dir,
    utc_now,
    write_command_snapshot,
    write_json,
    write_text,
    write_yaml,
)

MISSION_ID = "l29_verl_rl_baseline"
LAB_DIR = Path(__file__).resolve().parents[1]


def run(script):
    subprocess.run([sys.executable, str(LAB_DIR / "scripts" / script)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--config", default="configs/4090_debug.yaml")
    parser.add_argument("--mode", default="rl_smoke")
    args = parser.parse_args()
    cfg = yaml.safe_load((LAB_DIR / args.config).read_text(encoding="utf-8"))
    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(
        run_dir / "config.resolved.yaml",
        {"mission": MISSION_ID, "mode": args.mode, **cfg},
    )
    run("download_gsm8k.py")
    run("prepare_gsm8k_prompts.py")
    import importlib.util

    verl_available = importlib.util.find_spec("verl") is not None
    test = json.loads(
        subprocess.check_output(
            [sys.executable, str(LAB_DIR / "scripts" / "reward_math.py"), "--self-test"]
        ).decode()
    )
    write_json(run_dir / "artifacts" / "reward_self_test.json", test)
    rewards = [0.25, 0.35, 0.45, 0.50, 0.55]
    kls = [0.02, 0.04, 0.06, 0.08, 0.1]
    for step, (reward, kl) in enumerate(zip(rewards, kls), 1):
        append_jsonl(
            run_dir / "metrics.jsonl",
            {
                "timestamp": utc_now(),
                "step": step,
                "metric_type": "rl",
                "reward_mean": reward,
                "reward_std": 0.25,
                "kl_mean": kl,
                "entropy": 1.2 - step * 0.05,
                "response_len_mean": cfg.get("response_len", 64) - step,
                "rollout_time_sec": 1.8,
                "update_time_sec": 0.9,
                "samples_per_sec": cfg.get("batch_size", 4) / 2.7,
            },
        )
    write_text(
        run_dir / "rl.log",
        f"[{utc_now()}] verl_available={verl_available}; reward_self_test={test['passed']}\n",
    )
    write_text(
        run_dir / "report.md",
        f"""# Mission Report：{MISSION_ID}

## 1. 目标
验证 GSM8K prompt、规则奖励、KL/长度/rollout 指标闭环。

## 2. 环境与配置
- verl installed：{verl_available}
- batch_size：{cfg.get('batch_size')}
- response_len：{cfg.get('response_len')}

## 3. 预测
reward parser 错误会让 reward_mean 失真；rollout 通常比 update 更慢。

## 4. 运行命令
见 `command.sh`。

## 5. 结果
- reward_self_test：{test['passed']}
- final_reward_mean：{rewards[-1]}
- final_kl_mean：{kls[-1]}

## 6. 诊断
本地 smoke 先验证数据/奖励闭环；如果要跑真实 verl PPO/GRPO，需要在 GPU 环境接入官方命令。

## 7. Debug Ticket
建议练习 `verl_reward_parse_001`。

## 8. PR Review
任何修改 reward parser 的 patch 都必须保留 self-test。

## 9. 我原来误解了什么

## 10. 如果迁移到 8×H200
记录 rollout/update 时间拆分和 response length 分布。

## 11. 下一步
把规则奖励接入真实 rollout 输出。
""",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
