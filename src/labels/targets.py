from __future__ import annotations

from typing import Dict, Iterable, Sequence

import pandas as pd


DEFAULT_TARGET_MAP: Dict[str, str] = {
    "points": "event_points",
    "expected_goals": "expected_goals",
    "expected_assists": "expected_assists",
    "defensive_contribution": "defensive_contribution",
    "goals_conceded": "goals_conceded",
    "saves": "saves",
}


def add_horizon_labels(
    df: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 10, 15),
    target_map: Dict[str, str] | None = None,
    id_col: str = "player_id",
    gw_col: str = "gw",
    require_all: bool = True,
) -> pd.DataFrame:
    if target_map is None:
        target_map = DEFAULT_TARGET_MAP
    if id_col not in df.columns or gw_col not in df.columns:
        raise ValueError(f"Missing required columns: {id_col}, {gw_col}")

    df = df.copy()
    df = df.sort_values([id_col, gw_col])

    for target_name, src_col in target_map.items():
        if src_col not in df.columns:
            if require_all:
                raise ValueError(f"Missing target source column: {src_col}")
            continue
        for h in horizons:
            # Sum over next h GWs: t+1..t+h
            total = df.groupby(id_col, sort=False)[src_col].transform(
                lambda s, h=h: s.shift(-1)
                .rolling(window=h, min_periods=h)
                .sum()
                .shift(-(h - 1))
            )
            df[f"{target_name}_next_{h}_total"] = total
            df[f"{target_name}_next_{h}_per_week"] = total / float(h)

    return df


def labelable_mask(
    df: pd.DataFrame,
    max_finished_gw: int,
    horizons: Sequence[int],
    gw_col: str = "gw",
) -> pd.Series:
    if gw_col not in df.columns:
        raise ValueError(f"Missing column: {gw_col}")
    max_h = max(horizons) if horizons else 0
    return df[gw_col].astype(int) <= int(max_finished_gw - max_h)
