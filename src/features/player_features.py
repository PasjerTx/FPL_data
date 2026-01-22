from __future__ import annotations

from typing import Iterable, List, Sequence

import pandas as pd


DEFAULT_ROLLING_COLS = [
    "minutes",
    "event_points",
    "expected_goals",
    "expected_assists",
    "goals_scored",
    "assists",
    "defensive_contribution",
    "saves",
    "clean_sheets",
    "goals_conceded",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "bps",
    "bonus",
]


def _available_cols(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def add_per90_features(
    df: pd.DataFrame, cols: Iterable[str], minutes_col: str = "minutes"
) -> pd.DataFrame:
    df = df.copy()
    if minutes_col not in df.columns:
        return df
    mins = pd.to_numeric(df[minutes_col], errors="coerce").replace(0, pd.NA)
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[f"{c}_per90"] = (df[c] / mins) * 90.0
    return df


def add_player_rolling_features(
    df: pd.DataFrame,
    windows: Sequence[int] = (3, 5, 10),
    cols: Iterable[str] | None = None,
    per90: bool = True,
    include_current: bool = True,
    id_col: str = "player_id",
    gw_col: str = "gw",
    minutes_col: str = "minutes",
) -> pd.DataFrame:
    if cols is None:
        cols = DEFAULT_ROLLING_COLS
    df = df.copy()
    if id_col not in df.columns or gw_col not in df.columns:
        raise ValueError(f"Missing required columns: {id_col}, {gw_col}")

    if per90:
        roll_cols = _available_cols(df, cols)
        df = add_per90_features(df, roll_cols, minutes_col=minutes_col)
        roll_cols = roll_cols + [
            f"{c}_per90" for c in roll_cols if f"{c}_per90" in df.columns
        ]
    else:
        roll_cols = _available_cols(df, cols)

    # Ensure numeric dtypes for rolling operations
    for c in roll_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values([id_col, gw_col])

    def _rolling(series: pd.Series, window: int) -> pd.Series:
        if include_current:
            return series.rolling(window, min_periods=1).mean()
        return series.shift(1).rolling(window, min_periods=1).mean()

    new_cols = {}
    grouped = df.groupby(id_col, sort=False)
    for col in roll_cols:
        series = df[col]
        for w in windows:
            name = f"{col}_roll{w}"
            new_cols[name] = grouped[col].transform(lambda s, w=w: _rolling(s, w))

    # Trend features (3 vs 10) for key metrics when available
    trend_pairs = [
        ("event_points", 3, 10),
        ("expected_goals", 3, 10),
        ("expected_assists", 3, 10),
    ]
    for base_col, w_short, w_long in trend_pairs:
        c_short = f"{base_col}_roll{w_short}"
        c_long = f"{base_col}_roll{w_long}"
        if c_short in new_cols and c_long in new_cols:
            new_cols[f"{base_col}_trend_{w_short}_{w_long}"] = (
                new_cols[c_short] - new_cols[c_long]
            )

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df
