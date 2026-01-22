from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class FinishedGwIndex:
    finished_gws: List[int]
    max_finished_gw: int


def _to_bool_series(series: pd.Series) -> pd.Series:
    vals = series.astype(str).str.strip().str.lower()
    return vals.isin(["true", "1", "t", "yes"])


def compute_finished_gws(matches: pd.DataFrame) -> FinishedGwIndex:
    if "gw" in matches.columns:
        gw_col = "gw"
    else:
        gw_col = "gameweek"
    if gw_col not in matches.columns:
        raise ValueError("matches must include gw or gameweek column")
    if "finished" not in matches.columns:
        raise ValueError("matches must include finished column")

    tmp = matches[[gw_col, "finished"]].copy()
    tmp[gw_col] = tmp[gw_col].astype(int)
    tmp["finished_bool"] = _to_bool_series(tmp["finished"])

    finished_gws: List[int] = []
    for gw, g in tmp.groupby(gw_col, sort=True):
        if g["finished_bool"].all():
            finished_gws.append(int(gw))

    max_finished = max(finished_gws) if finished_gws else 0
    return FinishedGwIndex(finished_gws=finished_gws, max_finished_gw=max_finished)
