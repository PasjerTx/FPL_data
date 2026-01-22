from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


TARGET_LABELS = {
    "points": "expected_points",
    "expected_goals": "expected_goals",
    "expected_assists": "expected_assists",
    "defensive_contribution": "expected_defcon",
    "goals_conceded": "expected_goals_conceded",
    "saves": "expected_saves",
}


def _prediction_cols(targets: Sequence[str], horizons: Sequence[int]) -> List[str]:
    cols: List[str] = []
    for t in targets:
        for h in horizons:
            cols.append(f"{t}_next_{h}_total_pred")
            cols.append(f"{t}_next_{h}_per_week_pred")
    return cols


def _rename_predictions(
    df: pd.DataFrame, targets: Sequence[str], horizons: Sequence[int]
) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for t in targets:
        out_label = TARGET_LABELS.get(t, t)
        for h in horizons:
            total_col = f"{t}_next_{h}_total_pred"
            per_week_col = f"{t}_next_{h}_per_week_pred"
            if h == 1:
                rename_map[total_col] = f"{out_label}_next_gw"
            else:
                rename_map[total_col] = f"{out_label}_{h}_total"
                rename_map[per_week_col] = f"{out_label}_{h}_per_week"
    return df.rename(columns=rename_map)


def export_position_predictions(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    targets: Sequence[str],
    horizons: Sequence[int],
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
    pred_cols = [c for c in _prediction_cols(targets, horizons) if c in predictions.columns]

    outputs: Dict[str, Path] = {}
    for position in sorted(predictions["position"].dropna().unique()):
        pos_df = predictions[predictions["position"] == position].copy()
        # Sort by next 10 GW points if available
        if "points_next_10_total_pred" in pos_df.columns:
            pos_df = pos_df.sort_values("points_next_10_total_pred", ascending=False)
        elif "expected_points_10_total" in pos_df.columns:
            pos_df = pos_df.sort_values("expected_points_10_total", ascending=False)
        pos_df = pos_df[base_cols + pred_cols + uncertainty_cols]
        # Round prediction columns to 1 decimal
        for c in pred_cols:
            pos_df[c] = pd.to_numeric(pos_df[c], errors="coerce").round(1)
        pos_df = _rename_predictions(pos_df, targets, horizons)
        path = output_root / f"{position.lower()}_predictions.csv"
        pos_df.to_csv(path, index=False)
        outputs[position] = path

    return outputs
