from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


def build_availability_flags(
    pgw_all: pd.DataFrame,
    window_gw: int = 5,
    id_col: str = "player_id",
    gw_col: str = "gw",
    minutes_col: str = "minutes",
) -> pd.DataFrame:
    df = pgw_all.copy()
    if "id" in df.columns and id_col not in df.columns:
        df = df.rename(columns={"id": id_col})
    for col in [id_col, gw_col, minutes_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df[gw_col] = df[gw_col].astype(int)
    df = df.sort_values([id_col, gw_col])

    minutes = df[minutes_col].fillna(0)
    df["_dnp"] = (minutes <= 0).astype(int)

    df["missed_last_n"] = (
        df.groupby(id_col, sort=False)["_dnp"]
        .transform(lambda s: s.rolling(window_gw, min_periods=1).sum())
        .astype(int)
    )

    df["avg_minutes_last_n"] = df.groupby(id_col, sort=False)[minutes_col].transform(
        lambda s: s.rolling(window_gw, min_periods=1).mean()
    )

    if "starts" in df.columns:
        df["starts_last_n"] = df.groupby(id_col, sort=False)["starts"].transform(
            lambda s: s.rolling(window_gw, min_periods=1).sum()
        )

    # Direct availability indicators, if present
    if "status" in df.columns:
        df["status_flag"] = df["status"].astype(str).str.lower().ne("a")
    if "chance_of_playing_next_round" in df.columns:
        df["chance_flag"] = pd.to_numeric(
            df["chance_of_playing_next_round"], errors="coerce"
        ).fillna(100) < 100

    keep_cols = [
        id_col,
        gw_col,
        "missed_last_n",
        "avg_minutes_last_n",
        "starts_last_n",
        "status_flag",
        "chance_flag",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


def add_substitution_rates(
    base_df: pd.DataFrame,
    playermatchstats: pd.DataFrame,
    matches: pd.DataFrame,
    window_gw: int = 5,
    early_sub_min: int = 70,
    sub_on_min: int = 1,
    id_col: str = "player_id",
    gw_col: str = "gw",
) -> pd.DataFrame:
    if playermatchstats.empty or matches.empty:
        return base_df
    if "match_id" not in playermatchstats.columns or "match_id" not in matches.columns:
        return base_df

    pm = playermatchstats.copy()
    if "player_id" not in pm.columns:
        return base_df

    if "gw" not in matches.columns and "gameweek" in matches.columns:
        matches = matches.copy()
        matches["gw"] = matches["gameweek"]

    if "gw" not in pm.columns:
        pm = pm.merge(matches[["match_id", "gw"]], on="match_id", how="left")
        if "gw" in pm.columns:
            pm = pm.dropna(subset=["gw"])
    else:
        # if gw already present (from per-GW files), keep it
        if "gw_y" in pm.columns and "gw_x" in pm.columns:
            pm["gw"] = pm["gw_y"].fillna(pm["gw_x"])
    if "gw" not in pm.columns:
        return base_df
    pm = pm.dropna(subset=["gw"])
    pm["gw"] = pm["gw"].astype(int)

    if "start_min" not in pm.columns or "finish_min" not in pm.columns:
        return base_df

    pm["early_sub"] = (
        (pm["start_min"] <= sub_on_min)
        & (pm["finish_min"] < early_sub_min)
        & (pm["minutes_played"] > 0)
    ).astype(int)
    pm["sub_on"] = (pm["start_min"] > sub_on_min).astype(int)

    gw_rates = (
        pm.groupby([id_col, "gw"], as_index=False)
        .agg(early_sub_rate=("early_sub", "mean"), sub_on_rate=("sub_on", "mean"))
        .sort_values([id_col, "gw"])
    )

    for col in ["early_sub_rate", "sub_on_rate"]:
        gw_rates[f"{col}_last_n"] = gw_rates.groupby(id_col, sort=False)[col].transform(
            lambda s: s.rolling(window_gw, min_periods=1).mean()
        )

    out = base_df.merge(
        gw_rates[[id_col, "gw", "early_sub_rate_last_n", "sub_on_rate_last_n"]],
        left_on=[id_col, gw_col],
        right_on=[id_col, "gw"],
        how="left",
    )
    out = out.drop(columns=["gw_y"], errors="ignore").rename(columns={"gw_x": gw_col})
    return out
