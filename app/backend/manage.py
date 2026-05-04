from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backend.repository import dashboard_payload, list_quests  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Infra Quest 中文课程仓库管理器")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list-missions")
    subparsers.add_parser("dashboard")
    args = parser.parse_args()

    if args.command == "list-missions":
        for quest in list_quests():
            print(f'{quest["level"]}  {quest["id"]}  {quest["title"]}')
        return

    if args.command == "dashboard":
        print(json.dumps(dashboard_payload(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
