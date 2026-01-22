from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


def export_top_n_summaries(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    target: str = "points",
    horizons: Sequence[int] = (1, 5, 10, 15),
    top_n: int = 20,
    sort_horizon: int = 10,
) -> Dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    base_cols = [
        c
        for c in [
            "player_id",
            "first_name",
            "second_name",
            "web_name",
            "team_name",
            "position",
            "now_cost",
        ]
        if c in predictions.columns
    ]
    uncertainty_cols = [
        c
        for c in [
            "status_flag",
            "chance_flag",
            "missed_last_n",
            "avg_minutes_last_n",
            "starts_last_n",
            "early_sub_rate_last_n",
            "sub_on_rate_last_n",
        ]
        if c in predictions.columns
    ]

    pred_cols = []
    for h in horizons:
        pred_cols.append(f"{target}_next_{h}_total_pred")
        pred_cols.append(f"{target}_next_{h}_per_week_pred")
    pred_cols = [c for c in pred_cols if c in predictions.columns]

    sort_col = f"{target}_next_{sort_horizon}_total_pred"
    if sort_col not in predictions.columns:
        sort_col = pred_cols[0] if pred_cols else None

    outputs: Dict[str, Path] = {}
    for position in sorted(predictions["position"].dropna().unique()):
        pos_df = predictions[predictions["position"] == position].copy()
        if sort_col:
            pos_df = pos_df.sort_values(sort_col, ascending=False)
        pos_df = pos_df[base_cols + pred_cols + uncertainty_cols].head(top_n)
        for c in pred_cols:
            pos_df[c] = pd.to_numeric(pos_df[c], errors="coerce").round(1)
        path = output_root / f"{position.lower()}_top{top_n}.csv"
        pos_df.to_csv(path, index=False)
        outputs[position] = path

    if sort_col:
        overall = predictions.sort_values(sort_col, ascending=False)
        overall = overall[base_cols + pred_cols + uncertainty_cols].head(top_n)
        for c in pred_cols:
            overall[c] = pd.to_numeric(overall[c], errors="coerce").round(1)
        overall_path = output_root / f"overall_top{top_n}.csv"
        overall.to_csv(overall_path, index=False)
        outputs["overall"] = overall_path

    return outputs
