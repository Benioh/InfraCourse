from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
LABS = ROOT / "labs"
QUESTS = ROOT / "quests"
CURRICULUM = ROOT / "mini_infra" / "curriculum_map.yaml"

REQUIRED_DOCS = ("system_map.md", "lecture.md", "source_walkthrough.md")
OUTPUT_FILES = ("README.md", "debug_checklist.md", "source_reading_card.md")
CODE_EXTS = {".py", ".sh", ".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp", ".ts", ".tsx", ".js", ".mjs"}

EXTRA_SOURCE_PATHS = {
    "l07_dataset_megatron_bin": [
        "mini_infra/data/indexed_dataset.py",
        "github_repo/Megatron-LM/tools/preprocess_data.py",
        "github_repo/Megatron-LM/megatron/core/datasets/indexed_dataset.py",
    ],
    "l12_fsdp2_llama": [
        "github_repo/torchtitan/torchtitan/models/llama3/parallelize.py",
        "github_repo/torchtitan/torchtitan/trainer.py",
    ],
    "l14_pipeline_1f1b": [
        "mini_infra/megatron/core/pipeline_parallel/schedules.py",
        "github_repo/Megatron-LM/megatron/core/pipeline_parallel/schedules.py",
    ],
    "l16_resume_after_crash": [
        "mini_infra/megatron/training/checkpointing.py",
        "github_repo/Megatron-LM/megatron/training/checkpointing.py",
    ],
    "l22_flash_attn_v2_bench": [
        "mini_infra/gpu/microbench.py",
        "mini_infra/gpu/triton_softmax.py",
        "github_repo/triton/python/tutorials/02-fused-softmax.py",
    ],
    "l25_serve_eval_lm_eval": [
        "mini_infra/serving/benchmark.py",
        "github_repo/vllm/benchmarks/benchmark_serving.py",
    ],
    "l28_sft_qwen_alpaca": [
        "github_repo/sglang/python/sglang/lang/chat_template.py",
        "github_repo/sglang/python/sglang/srt/utils/hf_transformers/tokenizer.py",
    ],
    "l30_dpo_loss": [
        "mini_infra/rl/reward.py",
        "github_repo/slime/slime/utils/ppo_utils.py",
    ],
    "l34_grpo": [
        "mini_infra/rl/reward.py",
        "github_repo/slime/tests/test_chunked_gae.py",
        "github_repo/slime/slime/utils/ppo_utils.py",
    ],
}


class LiteralStr(str):
    pass


class Dumper(yaml.SafeDumper):
    pass


def _literal_representer(dumper: yaml.Dumper, data: LiteralStr) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


def _string_representer(dumper: yaml.Dumper, data: str) -> yaml.Node:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


Dumper.add_representer(LiteralStr, _literal_representer)
Dumper.add_representer(str, _string_representer)


AI_PHRASE_REPLACEMENTS = {
    "随着人工智能技术的快速发展": "在当前大模型系统里",
    "随着大模型的快速发展": "在大模型系统里",
    "随着……快速发展": "在系统规模变大时",
    "具有重要意义": "会影响工程判断",
    "发挥着至关重要的作用": "是关键路径上的约束",
    "为……提供了强有力的支撑": "支撑了这一层实现",
    "赋能": "支持",
    "全面提升": "改善",
    "极大地提高": "提高",
    "本文将深入探讨": "本讲会拆开",
    "综上所述": "小结",
    "值得注意的是": "需要看清的是",
    "不是，而是": "要改成",
}


def clean_text(text: str) -> str:
    for old, new in AI_PHRASE_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text.rstrip() + "\n"


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(clean_text(text), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(data, fh, Dumper=Dumper, allow_unicode=True, sort_keys=False, width=1000)


def lab_sort_key(lab_id: str) -> tuple[float, str]:
    match = re.match(r"l(\d+)(?:\.(\d+))?", lab_id)
    if not match:
        return (999.0, lab_id)
    major = int(match.group(1))
    minor = match.group(2)
    return (major + (int(minor) / 10 if minor else 0), lab_id)


def all_lab_ids() -> list[str]:
    return [
        p.name
        for p in sorted(LABS.glob("l*"), key=lambda p: lab_sort_key(p.name))
        if p.is_dir() and p.name != "l20_vllm_scheduler_kv"
    ]


def first_heading(path: Path) -> str:
    for line in read_text(path).splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.parent.name


def readme_without_generated_block(lab_id: str) -> str:
    text = read_text(LABS / lab_id / "README.md")
    text = re.sub(r"<!-- LECTURE_FIRST_START -->.*?<!-- LECTURE_FIRST_END -->", "", text, flags=re.S)
    return text.strip()


def readme_summary(lab_id: str) -> str:
    text = readme_without_generated_block(lab_id)
    lines = [line.rstrip() for line in text.splitlines()]
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("> ") and len(stripped) > 12:
            return stripped[2:].strip().strip("*")
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("```") or stripped.startswith("|"):
            continue
        if len(stripped) > 18:
            return stripped.strip("*")
    return ""


def readme_sections(lab_id: str) -> dict[str, str]:
    text = readme_without_generated_block(lab_id)
    sections: dict[str, list[str]] = {}
    current = ""
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current:
            sections[current].append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items() if "\n".join(value).strip()}


def task_summary(lab_id: str) -> str:
    text = read_text(LABS / lab_id / "patch" / "task.md")
    if not text:
        return ""
    wanted: list[str] = []
    capture = False
    for line in text.splitlines():
        if line.startswith("## 你要交付什么"):
            capture = True
            continue
        if capture and line.startswith("## "):
            break
        if capture:
            wanted.append(line)
    summary = "\n".join(wanted).strip()
    if not summary:
        summary = "\n".join(text.splitlines()[1:25]).strip()
    return summary[:1800]


def title_from_heading(heading: str, lab_id: str) -> tuple[str, str]:
    level = ""
    title = heading
    match = re.match(r"(L[0-9.]+)\s*[·:：-]\s*(.*)", heading)
    if match:
        level = match.group(1)
        title = match.group(2).strip()
    title = re.sub(r"（.*?）", "", title).strip()
    if "：" in title:
        title = title.split("：", 1)[0].strip()
    if ":" in title:
        title = title.split(":", 1)[0].strip()
    return title or lab_id, level


def infer_act(lab_id: str) -> str:
    n = int(re.match(r"l(\d+)", lab_id).group(1))
    if n == 1:
        return "第 0 章：入门与环境"
    if n <= 6:
        return "第 1 章：PyTorch、CUDA 与分布式基础"
    if n <= 18:
        return "第 2 章：训练系统、并行与数据"
    if n <= 27:
        return "第 3 章：推理服务"
    if n <= 34:
        return "第 4 章：SFT、偏好优化与 RL"
    return "第 5 章：多模态交付"


def infer_line(lab_id: str, act: str) -> str:
    if "环境" in act:
        return "Environment and evidence chain"
    if "推理" in act or any(key in lab_id for key in ("vllm", "sglang", "serve", "spec", "quant", "flash")):
        return "Serving systems"
    if "RL" in act or any(key in lab_id for key in ("rl", "dpo", "grpo", "slime", "rollout", "gae")):
        return "RLHF and rollout systems"
    if any(key in lab_id for key in ("data", "dataset", "multimodal")):
        return "Data pipeline"
    if any(key in lab_id for key in ("checkpoint", "resume")):
        return "Training reliability"
    if any(key in lab_id for key in ("pipeline", "fsdp", "parallel", "megatron", "moe", "torchtitan")):
        return "Training systems"
    return "AI Infra systems"


def template_name(lab_id: str, line: str) -> str:
    if "Environment" in line:
        return "env_report_template.md"
    if "Serving" in line:
        return "serving_metrics_template.md"
    if "RLHF" in line:
        return "rl_rollout_template.md"
    if "Data" in line:
        return "data_pipeline_template.md"
    if "reliability" in line:
        return "checkpoint_debug_template.md"
    if "l35" in lab_id:
        return "capstone_delivery_template.md"
    if "kernel" in lab_id or "flash" in lab_id or "memory" in lab_id:
        return "performance_metrics_template.md"
    if "eval" in lab_id:
        return "evaluation_report_template.md"
    return "training_step_template.md"


def patch_files(lab_id: str) -> dict[str, str]:
    patch_dir = LABS / lab_id / "patch"
    starter = next((p for p in sorted((patch_dir / "starter").glob("*.py")) if p.name != "__init__.py"), None)
    reference = next((p for p in sorted((patch_dir / "reference").glob("*.py")) if p.name != "__init__.py"), None)
    tests = next((p for p in sorted((patch_dir / "tests").glob("test_*.py"))), None)
    return {
        "task_md": f"labs/{lab_id}/patch/task.md",
        "starter_file": str(starter.relative_to(ROOT)) if starter else "",
        "reference_file": str(reference.relative_to(ROOT)) if reference else "",
        "tests_file": str(tests.relative_to(ROOT)) if tests else "",
    }


def count_tests(lab_id: str) -> int:
    tests_file = LABS / lab_id / "patch" / "tests" / "test_patch.py"
    text = read_text(tests_file)
    return len(re.findall(r"^def test_", text, flags=re.M))


def is_code_path(path: str) -> bool:
    if not path or path.endswith("/"):
        return False
    if path.startswith("docs/") or "/docs/" in path:
        return False
    if any(path.endswith(ext) for ext in (".md", ".yaml", ".yml", ".json", ".csv")):
        return False
    suffix = Path(path).suffix
    if suffix:
        return suffix in CODE_EXTS
    return Path(path).name in {"Makefile"}


def local_path(repo_path: str) -> Path:
    return ROOT / repo_path


def existing_code_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        if path and is_code_path(path) and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def source_paths_from_quest(data: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for item in data.get("source_reading") or []:
        if isinstance(item, dict):
            paths.append(str(item.get("repo_path", "")))
        elif isinstance(item, str):
            paths.append(item)
    for section in (data.get("lesson") or {}).get("sections") or []:
        for ref in section.get("source_refs") or []:
            if isinstance(ref, dict):
                paths.append(str(ref.get("repo_path", "")))
    return paths


def source_paths_from_curriculum(curriculum: dict[str, Any], lab_id: str) -> list[str]:
    labs = curriculum.get("labs") or {}
    info = labs.get(lab_id) or {}
    paths: list[str] = []
    for key in ("mini_module", "source_reading"):
        value = info.get(key)
        if isinstance(value, str):
            paths.append(value)
        elif isinstance(value, list):
            paths.extend(str(v) for v in value)
    return paths


def source_paths_from_lab(lab_id: str) -> list[str]:
    paths: list[str] = []
    paths.extend(EXTRA_SOURCE_PATHS.get(lab_id, []))
    patch = patch_files(lab_id)
    for key in ("starter_file", "reference_file", "tests_file"):
        if patch.get(key):
            paths.append(patch[key])
    for script in sorted((LABS / lab_id / "scripts").glob("*.py"))[:4]:
        paths.append(str(script.relative_to(ROOT)))
    return paths


def find_symbol_lines(path: Path) -> list[tuple[str, int, int]]:
    text = read_text(path)
    if not text:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        lines = text.splitlines()
        first = next((i + 1 for i, line in enumerate(lines) if line.strip() and not line.strip().startswith("#")), 1)
        return [("主入口", first, min(first + 8, len(lines)))]
    symbols: list[tuple[str, int, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end = getattr(node, "end_lineno", node.lineno)
            symbols.append((node.name, node.lineno, min(end, node.lineno + 12)))
        if len(symbols) >= 3:
            break
    if not symbols:
        lines = text.splitlines()
        first = next((i + 1 for i, line in enumerate(lines) if line.strip()), 1)
        symbols.append(("主入口", first, min(first + 8, len(lines))))
    return symbols


def focus_for_path(path: str) -> str:
    name = Path(path).name
    stem = Path(path).stem
    if name == "test_patch.py":
        return "测试用例和行为合同"
    if "starter" in path:
        return "学生需要补齐的最小实现"
    if "reference" in path:
        return "参考实现中的状态变化和边界处理"
    if stem in {"scheduler", "llm_engine", "kv_cache_manager"}:
        return "请求生命周期、调度状态和资源合同"
    if stem in {"training", "trainer", "train"}:
        return "训练 step、日志、checkpoint 和 metrics"
    if "checkpoint" in stem or "resume" in stem:
        return "保存、恢复、原子写入和一致性检查"
    if "radix" in stem or "cache" in stem:
        return "缓存键、命中、插入和释放"
    return f"{stem} 的主路径"


def title_for_path(path: str) -> str:
    if path.startswith("mini_infra/"):
        return "MiniInfra " + Path(path).stem
    if path.startswith("github_repo/"):
        parts = Path(path).parts
        repo = parts[1] if len(parts) > 1 else "真实源码"
        return f"{repo} {Path(path).stem}"
    if "/patch/starter/" in path:
        return "Patch starter"
    if "/patch/reference/" in path:
        return "Patch reference"
    if "/patch/tests/" in path:
        return "Patch tests"
    return Path(path).stem


def source_entry(path: str) -> dict[str, Any]:
    item: dict[str, Any] = {
        "title": title_for_path(path),
        "repo_path": path,
        "focus": focus_for_path(path),
    }
    lpath = local_path(path)
    symbols = find_symbol_lines(lpath)
    if symbols:
        highlights = []
        for symbol, start, end in symbols[:2]:
            highlights.append(
                {
                    "title": f"读 {symbol}",
                    "lines": [start, end],
                    "note": f"- L{start}-L{end}: 先看这一段如何建立 `{focus_for_path(path)}`。读完后要能说清输入、状态变化、输出和失败边界。",
                }
            )
        item["highlights"] = highlights
    return item


def sanitize_source_reading(data: dict[str, Any], fallback_paths: list[str]) -> None:
    existing: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in data.get("source_reading") or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("repo_path", ""))
        if not is_code_path(path) or path in seen:
            continue
        seen.add(path)
        if not item.get("highlights"):
            generated = source_entry(path)
            item.setdefault("focus", generated["focus"])
            if generated.get("highlights"):
                item["highlights"] = generated["highlights"]
        existing.append(item)
    for path in fallback_paths:
        if path not in seen and is_code_path(path):
            seen.add(path)
            existing.append(source_entry(path))
    data["source_reading"] = existing[:8]

    for section in (data.get("lesson") or {}).get("sections") or []:
        refs = []
        for ref in section.get("source_refs") or []:
            if isinstance(ref, dict) and is_code_path(str(ref.get("repo_path", ""))):
                refs.append(ref)
        if refs:
            section["source_refs"] = refs[:5]
        elif "source_refs" in section:
            section.pop("source_refs")


def lesson_docs(lab_id: str) -> list[dict[str, str]]:
    return [
        {
            "title": "系统地图",
            "path": f"labs/{lab_id}/system_map.md",
            "description": "本讲在章节中的位置、局部系统图、概念依赖和易混点。",
        },
        {
            "title": "主讲义",
            "path": f"labs/{lab_id}/lecture.md",
            "description": "从真实问题、核心机制、源码抽象讲到 lab 验收和生产排查。",
        },
        {
            "title": "源码带读讲义",
            "path": f"labs/{lab_id}/source_walkthrough.md",
            "description": "按主路径组织的源码阅读路线，说明每步要读什么和跳过什么。",
        },
    ]


def chapter_labs(lab_id: str, metadata: dict[str, dict[str, Any]]) -> list[str]:
    act = metadata[lab_id]["act"]
    return [lid for lid in all_lab_ids() + ["l20_vllm_scheduler_kv"] if metadata.get(lid, {}).get("act") == act]


def build_metadata(curriculum: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    ids = sorted({*all_lab_ids(), "l20_vllm_scheduler_kv"}, key=lab_sort_key)
    for lab_id in ids:
        lab_dir = LABS / lab_id
        q = load_yaml(QUESTS / f"{lab_id}.yaml")
        heading = first_heading(lab_dir / "README.md")
        parsed_title, parsed_level = title_from_heading(heading, lab_id)
        cur_info = (curriculum.get("labs") or {}).get(lab_id) or {}
        title = q.get("title") or parsed_title
        level = q.get("level") or parsed_level or lab_id.upper()
        act = q.get("act") or infer_act(lab_id)
        line = infer_line(lab_id, act)
        source_paths = existing_code_paths(
            source_paths_from_quest(q) + source_paths_from_curriculum(curriculum, lab_id) + source_paths_from_lab(lab_id)
        )
        patch = q.get("patch") or {}
        pf = patch_files(lab_id)
        patch.setdefault("description", f"实现 {title} 的最小行为合同，并用测试验证关键边界。")
        patch.setdefault("task_md", pf["task_md"])
        if pf["starter_file"]:
            patch.setdefault("starter_file", pf["starter_file"])
        if pf["reference_file"]:
            patch.setdefault("reference_file", pf["reference_file"])
        patch.setdefault("test_command", f"make patch-test M={lab_id}")
        patch.setdefault("status_file", f"labs/{lab_id}/patch/.last_run.json")
        patch.setdefault("test_count", count_tests(lab_id))
        patch.setdefault("test_kind", "最小行为合同验证")
        metadata[lab_id] = {
            "id": lab_id,
            "title": title,
            "level": level,
            "act": act,
            "line": line,
            "role": q.get("role") or "AI Infra 学员",
            "priority": q.get("priority") or "核心",
            "frameworks": q.get("frameworks") or infer_frameworks(lab_id, title, source_paths),
            "notebooks": q.get("notebooks") or cur_info.get("notebooks") or [],
            "mini_infra_targets": q.get("mini_infra_targets") or normalize_list(cur_info.get("mini_module")),
            "source_paths": source_paths,
            "patch": patch,
            "quest": q,
            "curriculum": cur_info,
        }
    return metadata


def normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value]
    return []


def infer_frameworks(lab_id: str, title: str, paths: list[str]) -> list[str]:
    haystack = " ".join([lab_id, title, *paths]).lower()
    frameworks = ["MiniInfra"]
    for label, key in [
        ("PyTorch", "torch"),
        ("Megatron-LM", "megatron"),
        ("vLLM", "vllm"),
        ("SGLang", "sglang"),
        ("SLiME", "slime"),
        ("verl", "verl"),
        ("TorchTitan", "torchtitan"),
        ("Triton", "triton"),
        ("HuggingFace", "qwen"),
    ]:
        if key.lower() in haystack and label not in frameworks:
            frameworks.append(label)
    return frameworks


def learning_goals(meta: dict[str, Any]) -> list[str]:
    q_goals = meta["quest"].get("learning_goals")
    if q_goals:
        return [str(goal) for goal in q_goals[:5]]
    title = meta["title"]
    line = meta["line"]
    return [
        f"解释 {title} 解决的真实 {line} 问题。",
        "画出入口、核心状态、源码主路径和输出 artifact 的关系。",
        "说清关键机制的输入、中间状态、输出、代价和边界。",
        "读懂本讲 MiniInfra/真实源码中的主函数和关键字段。",
        "完成 patch，并用测试、drill 或 notebook 验收最小合同。",
    ]


def one_sentence(meta: dict[str, Any]) -> str:
    inc = meta["curriculum"].get("increment")
    if inc:
        return str(inc).replace("\n", " ")
    opening = (meta["quest"].get("lesson") or {}).get("opening", "")
    if "本讲围绕" not in opening or "最小 patch 验收" not in opening:
        for line in opening.splitlines():
            line = line.strip("# ").strip()
            if re.match(r"^L[0-9.]+[：:]", line):
                continue
            if len(line) >= 20 and not line.startswith("第 "):
                return line
    summary = readme_summary(meta["id"])
    if summary:
        return summary
    return f"本讲围绕 {meta['title']}，把系统问题、关键机制、源码主路径和最小 patch 验收串成一条可复查链路。"


def command_for_drill(lab_id: str) -> str:
    scripts = sorted((LABS / lab_id / "scripts").glob("run*.sh"))
    if scripts:
        return f"bash {scripts[0].relative_to(ROOT)}"
    py_scripts = sorted((LABS / lab_id / "scripts").glob("run*.py"))
    if py_scripts:
        return f"python {py_scripts[0].relative_to(ROOT)}"
    if (LABS / lab_id / "run.sh").exists():
        return f"bash labs/{lab_id}/run.sh"
    return f"make smoke M={lab_id}"


def update_readme(lab_id: str, meta: dict[str, Any]) -> None:
    path = LABS / lab_id / "README.md"
    original = read_text(path)
    if not original:
        original = f"# {meta['level']} · {meta['title']}\n"
    lines = original.splitlines()
    heading = lines[0] if lines and lines[0].startswith("# ") else f"# {meta['level']} · {meta['title']}"
    rest = "\n".join(lines[1:]).lstrip()
    block = f"""<!-- LECTURE_FIRST_START -->

{one_sentence(meta)}

## 学习路线

建议按下面顺序走，先把系统讲通，再写 patch。

1. 读 [system_map.md](system_map.md)：确认本讲在 `{meta['act']}` 和 `{meta['line']}` 主线里的位置。
2. 读 [lecture.md](lecture.md)：从真实问题、核心概念、机制和源码抽象进入。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按主路径阅读 MiniInfra、patch reference 和真实源码。
4. 跑 notebook：{', '.join(f'[{Path(n).name}](../../{n})' for n in meta['notebooks']) if meta['notebooks'] else '本讲没有强依赖 notebook，优先跑 drill 或 smoke。'}
5. 做 quiz：确认概念、边界和排查顺序。
6. 做 patch：实现最小行为合同并通过测试。
7. 跑 drill：用脚本或 smoke 命令观察指标和 artifact。
8. 填写 [outputs/{template_name(lab_id, meta['line'])}](outputs/{template_name(lab_id, meta['line'])})，沉淀复盘结论。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | {meta['line']} |
| 它解决什么问题 | {one_sentence(meta)} |
| 它连接哪些源码 | {', '.join(f'`{p}`' for p in meta['source_paths'][:4]) or 'patch reference、scripts 和 MiniInfra 主路径'} |
| lab 检验什么 | {meta['patch'].get('description', '').replace(chr(10), ' ')} |

## 你会学到什么

""" + "\n".join(f"- {goal}" for goal in learning_goals(meta)) + f"""

## Patch 闭环

```bash
cat labs/{lab_id}/patch/task.md
{f"$EDITOR {meta['patch'].get('starter_file')}" if meta['patch'].get('starter_file') else '# 打开 patch/starter 中的待实现文件'}
make patch-test M={lab_id}
```

drill 或 smoke：

```bash
{command_for_drill(lab_id)}
```

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 面向真实问题的排查顺序 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习源码主路径和关键结论 |
| [outputs/{template_name(lab_id, meta['line'])}](outputs/{template_name(lab_id, meta['line'])}) | 记录本讲指标、现象、判断和下一步动作 |

<!-- LECTURE_FIRST_END -->
"""
    rest = re.sub(r"<!-- LECTURE_FIRST_START -->.*?<!-- LECTURE_FIRST_END -->\n?", "", rest, flags=re.S).lstrip()
    write_text(path, f"{heading}\n\n{block}\n{rest}".rstrip() + "\n")


def generate_system_map(lab_id: str, meta: dict[str, Any], metadata: dict[str, dict[str, Any]]) -> None:
    chapter = sorted(chapter_labs(lab_id, metadata), key=lab_sort_key)
    route = "\n".join(
        f"{'-> ' if lid == lab_id else '   '}{metadata[lid]['level']} {metadata[lid]['title']}"
        for lid in chapter
    )
    sources = meta["source_paths"][:6]
    source_lines = "\n".join(f"  -> {path}" for path in sources) or "  -> patch/starter\n  -> patch/reference\n  -> patch/tests"
    text = f"""# 系统地图：{meta['title']}

{one_sentence(meta)}

## 1. {meta['act']} 路线图

```text
{route}
```

本讲在这条路线里负责把 `{meta['title']}` 的系统边界讲清楚。前一讲提供背景或输入，后一讲会复用这里建立的状态、指标或 artifact。

## 2. 本讲局部系统图

```text
Problem / workload
  |
  v
Notebook or drill
  |
  v
MiniInfra / patch reference
{source_lines}
  |
  v
Metrics, logs, checkpoints or outputs
  |
  v
Debug checklist and reusable report
```

这张图的重点是边界。notebook 和 drill 用来制造可观察现象；MiniInfra 和 reference 保留主路径；真实源码告诉你生产系统多出的配置、并发、容错和性能分支。

## 3. 概念依赖图

```text
真实问题
  -> 输入和约束
  -> 核心状态
  -> 机制步骤
  -> 指标或 artifact
  -> 源码落点
  -> patch 最小合同
  -> 生产排查路径
```

学习时先确认输入是什么，再看状态如何变化，最后才写 patch。这样不会把局部实现误当成整节课。

## 4. 本讲和 lab 的关系

```text
讲授主体：
  问题背景 + 系统位置 + 机制链路 + 源码主路径

演示材料：
  notebook / scripts / MiniInfra smoke

Lab 验收：
  {meta['patch'].get('description', '').replace(chr(10), ' ')}

可复用产物：
  outputs/debug_checklist.md
  outputs/source_reading_card.md
  outputs/{template_name(lab_id, meta['line'])}
```

lab 是出口验收。做 patch 前，学生应该能先解释本讲状态如何从输入走到输出，再说明测试为什么覆盖这个最小合同。

## 5. 易混点总表

| 易混点 | 正确理解 |
|---|---|
| patch 通过等于掌握整节课 | patch 只证明最小合同，系统定位、源码主路径和排查产物也要能讲清 |
| notebook 现象等于生产结论 | notebook 用来建立机制直觉，生产结论还要看配置、输入规模和真实指标 |
| 单个源码文件就是完整系统 | 真实系统通常还包含入口、状态管理、指标、错误处理和资源释放 |
| 性能数字可以脱离条件比较 | 必须同时记录输入规模、硬件、并发、dtype、版本和比较对象 |
| artifact 存在就说明成功 | artifact 要能支撑结论，缺字段或缺命令快照时只能算弱证据 |

## 6. 本讲结束后的能力标准

1. 能用一张图说明 {meta['title']} 在 `{meta['line']}` 主线中的位置。
2. 能解释核心机制的输入、中间状态、输出、代价和边界。
3. 能沿着 `source_walkthrough.md` 找到至少三个源码落点，并说出每段代码的结论。
4. 能完成 patch，并说明测试覆盖了哪些行为合同。
5. 能用 `outputs/debug_checklist.md` 对同类真实问题给出有顺序的排查路径。
"""
    write_text(LABS / lab_id / "system_map.md", text)


def generate_lecture(lab_id: str, meta: dict[str, Any]) -> None:
    lesson = meta["quest"].get("lesson") or {}
    opening = lesson.get("opening")
    sections = lesson.get("sections") or []
    parts: list[str] = [f"# {meta['level']}：{meta['title']}", ""]
    if opening:
        parts.append(re.sub(r"^# .*$", "", str(opening), flags=re.M).strip())
    else:
        parts.append(one_sentence(meta))
    parts.extend(["", "## 1. 本讲目标", ""])
    parts.extend(f"- {goal}" for goal in learning_goals(meta))
    parts.extend(
        [
            "",
            "## 2. 问题背景和系统位置",
            "",
            f"本讲属于 `{meta['line']}` 主线。要解决的问题是：{one_sentence(meta)}",
            "",
            "学习时先把系统位置放稳：输入从 workload、配置或请求进入，经过 MiniInfra 或真实框架的核心状态，再落到 metrics、日志、checkpoint、输出文件或服务响应。patch 只检查其中一个最小合同。",
            "",
        ]
    )
    if sections:
        start = 3
        for idx, section in enumerate(sections, start=start):
            title = section.get("title", f"机制 {idx}")
            parts.extend([f"## {idx}. {title}", ""])
            for key in ("plain_explanation", "mental_model", "why_it_matters"):
                value = section.get(key)
                if value:
                    label = {"plain_explanation": "", "mental_model": "**直观理解：**", "why_it_matters": "**为什么要学：**"}[key]
                    if label:
                        parts.extend([label, ""])
                    parts.extend([str(value).strip(), ""])
            confusions = section.get("common_confusions") or []
            if confusions:
                parts.extend(["**常见误解：**", ""])
                parts.extend(f"- {item}" for item in confusions)
                parts.append("")
            questions = section.get("checkpoint_questions") or []
            if questions:
                parts.extend(["**自检问题：**", ""])
                parts.extend(f"- {item}" for item in questions)
                parts.append("")
    else:
        sections = readme_sections(lab_id)
        task = task_summary(lab_id)
        parts.extend(
            [
                "## 3. 核心机制",
                "",
                "本讲的核心机制按四步读：输入是什么，状态在哪里保存，输出如何被消费，失败时留下什么证据。",
                "",
            ]
        )
        if sections.get("为什么这关重要"):
            parts.extend(["README 给出的工程动机：", "", sections["为什么这关重要"].strip(), ""])
        if task:
            parts.extend(["patch task 中定义的最小合同：", "", task, ""])
        parts.extend(
            [
                "下面这几个源码节点承担主路径：",
                "",
            ]
        )
        parts.extend(f"- `{path}`：{focus_for_path(path)}。" for path in meta["source_paths"][:6])
        parts.append("")
    parts.extend(
        [
            "## Lab 验收边界",
            "",
            f"本讲 patch 命令：`make patch-test M={lab_id}`。",
            "",
            f"patch 验收的是：{meta['patch'].get('description', '').replace(chr(10), ' ')}",
            "",
            "这不是整节课的全部。测试通过后，还要能把测试中的输入、状态变化、异常边界和真实源码主路径对上。",
            "",
            "## 生产排查入口",
            "",
            "遇到同类问题时，按下面顺序排查：",
            "",
            "1. 先确认 workload、配置、版本、硬件和随机种子。",
            "2. 再看入口日志、核心状态、指标和 artifact 是否完整。",
            "3. 然后沿源码主路径定位状态变化发生在哪一层。",
            "4. 最后比较 patch 或 MiniInfra 的最小合同，判断真实系统多出的复杂度来自哪里。",
            "",
            f"课后使用 `outputs/{template_name(lab_id, meta['line'])}` 记录一次完整复盘。",
        ]
    )
    write_text(LABS / lab_id / "lecture.md", "\n".join(parts))


def generate_source_walkthrough(lab_id: str, meta: dict[str, Any]) -> None:
    paths = meta["source_paths"][:8]
    parts = [
        f"# 源码带读：{meta['title']}",
        "",
        "这份带读按主路径组织。读源码时不要从文件顶部一路滚到底，先按下面步骤建立调用链，再回头看生产分支。",
        "",
        "## 0. 源码地图",
        "",
        "```text",
    ]
    if paths:
        parts.extend(paths)
    else:
        parts.extend([f"labs/{lab_id}/patch/starter", f"labs/{lab_id}/patch/reference", f"labs/{lab_id}/patch/tests"])
    parts.extend(["```", ""])
    for idx, path in enumerate(paths, start=1):
        lpath = local_path(path)
        symbols = find_symbol_lines(lpath)
        sym_text = "、".join(symbol for symbol, _, _ in symbols[:3]) if symbols else "主入口"
        parts.extend(
            [
                f"## {idx}. {title_for_path(path)}",
                "",
                f"文件：`{path}`",
                "",
                f"重点看：{sym_text}。",
                "",
                f"这一段要回答：{focus_for_path(path)} 的输入是什么，状态在哪里改变，输出给谁消费，失败时应该留下什么证据。",
                "",
            ]
        )
        if symbols:
            parts.append("建议阅读顺序：")
            parts.append("")
            for symbol, start, end in symbols[:3]:
                parts.append(f"- L{start}-L{end}: `{symbol}`。读完后写一句结论，说明它在本讲机制里承担的角色。")
            parts.append("")
        parts.extend(
            [
                "可以先跳过：",
                "",
                "- 与本讲主合同无关的 CLI 参数解析、日志格式化、测试夹具和兼容性分支。",
                "- 真实框架里暂时看不懂的性能特化分支，先记下入口，等主路径闭合后再回来看。",
                "",
            ]
        )
    parts.extend(
        [
            "## 读完后的自检问题",
            "",
            f"1. 你能否从入口画到 `{meta['title']}` 的核心状态？",
            "2. 哪个文件负责把输入转成内部状态，哪个文件负责输出或 artifact？",
            "3. patch 测试覆盖了哪些边界，没有覆盖哪些生产复杂度？",
            "4. 如果真实系统指标异常，你会先看哪三个状态或日志？",
        ]
    )
    write_text(LABS / lab_id / "source_walkthrough.md", "\n".join(parts))


def generate_outputs(lab_id: str, meta: dict[str, Any]) -> None:
    out_dir = LABS / lab_id / "outputs"
    template = template_name(lab_id, meta["line"])
    write_text(
        out_dir / "README.md",
        f"""# 课后产物：{meta['title']}

本目录存放这讲课后可以继续复用的材料。它们不是提交作业的格式，而是以后排查同类问题时可以直接拿来用的记录模板。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 按顺序定位 {meta['title']} 相关问题 |
| `source_reading_card.md` | 快速回忆源码主路径和关键结论 |
| `{template}` | 记录一次 drill、benchmark、训练或服务复盘 |

建议每次跑完 patch、drill 或真实任务后，把命令、配置、输入规模、关键指标和结论写进模板。没有这些证据时，不要把局部现象写成生产结论。
""",
    )
    write_text(
        out_dir / "debug_checklist.md",
        f"""# Debug Checklist：{meta['title']}

## 1. 先固定现场

- 记录命令、配置文件、git commit、Python 环境、硬件、输入数据和随机种子。
- 保存 stdout/stderr、metrics、日志、checkpoint 或服务响应，不只保存截图。
- 明确这是 patch-test、notebook、drill、smoke，还是真实框架运行。

## 2. 判断问题在哪一层

| 层 | 要看什么 | 可能结论 |
|---|---|---|
| 输入 | 数据、请求、prompt、shape、配置 | 输入规模或格式已经偏离预期 |
| 状态 | queue、batch、rank、cache、optimizer、checkpoint | 核心状态没有按机制推进 |
| 执行 | kernel、forward/backward、scheduler、worker | 执行路径慢、跳过或失败 |
| 输出 | metrics、artifact、loss、latency、accuracy | 结果无法支撑当前判断 |

## 3. 沿源码主路径复查

""" + "\n".join(f"- `{path}`：{focus_for_path(path)}。" for path in meta["source_paths"][:6]) + f"""

## 4. 常见错误判断

- 只看一个平均指标，没有拆输入规模、阶段和资源条件。
- 只确认文件存在，没有验证文件内容是否能恢复状态或支撑结论。
- 把 MiniInfra 的简化合同直接写成真实框架全部行为。
- patch 通过后没有跑 drill 或 notebook，缺少系统层现象。

## 5. 结束条件

- 问题能被一个最小命令复现。
- 关键状态和指标已经落盘。
- 源码主路径中能指出状态在哪里产生、改变和释放。
- 结论写进 `{template}`，并包含下一步动作。
""",
    )
    write_text(
        out_dir / "source_reading_card.md",
        f"""# Source Reading Card：{meta['title']}

## 主路径

""" + "\n".join(f"{idx}. `{path}`：{focus_for_path(path)}。" for idx, path in enumerate(meta["source_paths"][:8], start=1)) + """

## 阅读方法

1. 先看入口函数或公开 API，确认输入对象。
2. 再看核心状态字段，确认状态在哪里保存。
3. 继续看状态推进或资源分配逻辑，确认输出给谁消费。
4. 最后看测试，确认哪些边界被验收。

## 自检

- 我能否说清每个源码文件在系统图里的位置？
- 我能否指出 patch reference 和真实源码的同构关系？
- 我能否解释哪些分支可以先跳过，哪些分支会改变语义？
""",
    )
    write_text(
        out_dir / template,
        f"""# {meta['title']} 复盘模板

## Run 信息

- 日期：
- 机器 / GPU：
- 命令：
- 配置文件：
- git commit：
- 数据或 workload：

## 预期

- 这次运行要验证的机制：
- 比较对象：
- 成功标准：

## 观察指标

| 指标或 artifact | 数值 / 路径 | 解释 |
|---|---|---|
|  |  |  |

## 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
|  |  |  |

## 结论

- 本次能证明什么：
- 不能证明什么：
- 下一步要改的配置、代码或实验：
""",
    )


def build_default_lesson(meta: dict[str, Any]) -> dict[str, Any]:
    title = meta["title"]
    lab_id = meta["id"]
    sections = readme_sections(lab_id)
    summary = readme_summary(lab_id) or one_sentence(meta)
    task = task_summary(lab_id)
    motivation = sections.get("为什么这关重要") or sections.get("Drill") or sections.get("Eval") or sections.get("SFT smoke") or ""
    delivery = task or meta["patch"].get("description", "")
    return {
        "opening": LiteralStr(
            f"# {meta['level']}：{title}\n\n{summary}\n\n本讲按系统地图、主讲义、源码带读、notebook、quiz、patch、drill 和 outputs 的顺序组织。patch 只验收最小合同，讲授重点放在真实问题、机制链路、源码位置和复盘材料。"
        ),
        "sections": [
            {
                "title": "课程定位和验收边界",
                "plain_explanation": LiteralStr(
                    f"{title} 属于 `{meta['line']}` 主线。学习时先确认它解决哪个真实问题，再看输入、状态、输出和 artifact。patch 的作用是检查最小行为合同，不覆盖所有生产分支。\n\nREADME 中的一句话定位是：{summary}"
                ),
                "mental_model": "先把系统画成入口、状态、执行、输出四层，再把源码文件放到对应层。",
                "why_it_matters": "边界清楚后，学生能把测试失败、指标异常和源码状态联系起来，而不是只猜 patch。",
                "checkpoint_questions": [
                    "本讲解决的真实问题是什么？",
                    "patch-test 和 drill 分别证明什么？",
                ],
            },
            {
                "title": "核心机制",
                "plain_explanation": LiteralStr(
                    "这一讲的机制按四个问题读：输入是什么，核心状态是什么，输出是什么，失败边界是什么。每个概念都要能说清定义、直觉、代价和适用条件。\n\n"
                    + (f"课程动机或演练要求：\n{motivation}" if motivation else f"最小交付合同：\n{delivery}")
                ),
                "mental_model": "把机制看成一条状态转移链，每一步都必须能落到源码和指标。",
                "why_it_matters": "真实框架通常把同一个机制拆在多个文件里，先抓状态链才能读懂生产分支。",
                "checkpoint_questions": [
                    "核心状态在哪里创建、更新和释放？",
                    "哪些指标能证明机制按预期运行？",
                ],
            },
            {
                "title": "源码主路径",
                "plain_explanation": LiteralStr(
                    "源码阅读先从 MiniInfra 或 patch reference 进入，再对照真实框架。不要先追全部分支，先确认主函数、状态字段、错误处理和输出对象。"
                ),
                "mental_model": "MiniInfra 是缩小图，真实源码是放大图；两者共享主合同，但真实源码会多出并发、性能和容错分支。",
                "why_it_matters": "读源码的目标是解释系统行为，不是背文件名。",
                "source_refs": [source_entry(path) for path in meta["source_paths"][:3]],
                "checkpoint_questions": [
                    "哪些文件是主路径，哪些文件只提供测试或脚手架？",
                    "真实源码相比 patch reference 多了哪些生产复杂度？",
                ],
            },
            {
                "title": "Lab 和复盘",
                "plain_explanation": LiteralStr(
                    f"本讲 lab 使用 `{meta['patch'].get('test_command')}` 验收。通过后还要跑 drill 或 smoke，并把命令、配置、指标和判断写入 `outputs/{template_name(meta['id'], meta['line'])}`。\n\npatch task 摘要：\n{delivery}"
                ),
                "mental_model": "patch 是出口考试，outputs 是以后排查问题时能复用的记录。",
                "why_it_matters": "没有可复查 artifact，实验结论很难迁移到真实项目。",
                "checkpoint_questions": [
                    "测试覆盖了哪些最小合同？",
                    "复盘模板里必须记录哪些条件？",
                ],
            },
        ],
    }


def update_or_create_quest(lab_id: str, meta: dict[str, Any]) -> None:
    path = QUESTS / f"{lab_id}.yaml"
    data = load_yaml(path)
    if not data:
        data = {
            "id": lab_id,
            "title": meta["title"],
            "level": meta["level"],
            "act": meta["act"],
            "role": meta["role"],
            "priority": meta["priority"],
            "estimated_minutes": 120,
            "no_gpu_friendly": True,
            "frameworks": meta["frameworks"],
            "patch": meta["patch"],
            "mini_infra_targets": meta["mini_infra_targets"],
            "notebooks": meta["notebooks"],
            "gpu_modes": {
                "local": "CPU 或单卡 smoke",
                "cluster": "按 configs 中的集群模板验证",
            },
            "lesson": build_default_lesson(meta),
        }
    else:
        data.setdefault("id", lab_id)
        data.setdefault("title", meta["title"])
        data.setdefault("level", meta["level"])
        data.setdefault("act", meta["act"])
        data.setdefault("role", meta["role"])
        data.setdefault("priority", meta["priority"])
        data.setdefault("frameworks", meta["frameworks"])
        data.setdefault("patch", meta["patch"])
        data.setdefault("mini_infra_targets", meta["mini_infra_targets"])
        data.setdefault("notebooks", meta["notebooks"])
        lesson = data.get("lesson") or {}
        opening = str(lesson.get("opening", ""))
        if "本讲围绕" in opening and "最小 patch" in opening:
            data["lesson"] = build_default_lesson(meta)
        else:
            data.setdefault("lesson", build_default_lesson(meta))
    data["lesson_docs"] = lesson_docs(lab_id)
    fallback_paths = meta["source_paths"]
    sanitize_source_reading(data, fallback_paths)
    dump_yaml(path, data)


def main() -> None:
    curriculum = load_yaml(CURRICULUM)
    metadata = build_metadata(curriculum)
    for lab_id in all_lab_ids():
        meta = metadata[lab_id]
        update_readme(lab_id, meta)
        update_or_create_quest(lab_id, meta)
        meta["quest"] = load_yaml(QUESTS / f"{lab_id}.yaml")
        generate_system_map(lab_id, meta, metadata)
        generate_lecture(lab_id, meta)
        generate_source_walkthrough(lab_id, meta)
        generate_outputs(lab_id, meta)


if __name__ == "__main__":
    main()
