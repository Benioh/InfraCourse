from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboards.parse_metrics import read_metrics


def main() -> None:
    st.set_page_config(page_title="Infra Quest 指标看板", layout="wide")
    st.title("Infra Quest 指标看板")
    rows = read_metrics(Path(__file__).resolve().parents[1])
    st.write(f"已加载 {len(rows)} 行指标")
    st.dataframe(rows, use_container_width=True)


if __name__ == "__main__":
    main()
