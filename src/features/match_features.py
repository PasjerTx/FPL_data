from __future__ import annotations

from typing import Iterable, List, Sequence

import pandas as pd


def _ensure_gw(pm: pd.DataFrame, matches: pd.DataFrame | None = None) -> pd.DataFrame:
    out = pm.copy()
    if "gw" in out.columns:
        return out
    if matches is None or matches.empty or "match_id" not in out.columns:
        return out
    m = matches.copy()
    if "gw" not in m.columns and "gameweek" in m.columns:
        m["gw"] = m["gameweek"]
    if "gw" not in m.columns:
        return out
    out = out.merge(m[["match_id", "gw"]], on="match_id", how="left")
    return out


def build_playermatchstats_features(
    playermatchstats: pd.DataFrame,
    matches: pd.DataFrame | None = None,
    agg: Sequence[str] = ("sum", "mean"),
) -> pd.DataFrame:
    if playermatchstats.empty:
        return pd.DataFrame()

    pm = _ensure_gw(playermatchstats, matches)
    if "player_id" not in pm.columns or "gw" not in pm.columns:
        return pd.DataFrame()

    pm = pm.copy()
    pm["gw"] = pm["gw"].astype(int)

    numeric_cols = pm.select_dtypes(include=["number", "bool"]).columns
    exclude = {"player_id", "match_id", "gw", "start_min", "finish_min"}
    cols = [c for c in numeric_cols if c not in exclude]
    if not cols:
        return pd.DataFrame()

    grouped = pm.groupby(["player_id", "gw"], as_index=False)
    frames: List[pd.DataFrame] = []
    if "sum" in agg:
        s = grouped[cols].sum()
        s = s.rename(columns={c: f"pm_sum_{c}" for c in cols})
        frames.append(s)
    if "mean" in agg:
        m = grouped[cols].mean()
        m = m.rename(columns={c: f"pm_mean_{c}" for c in cols})
        frames.append(m)

    if not frames:
        return pd.DataFrame()

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["player_id", "gw"], how="left")
    return out
