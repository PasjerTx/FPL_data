from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


def _ensure_gw_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "gw" not in out.columns and "gameweek" in out.columns:
        out["gw"] = out["gameweek"]
    return out


def build_team_fixture_difficulty(
    fixtures: pd.DataFrame,
    teams_dim: pd.DataFrame | None = None,
    horizons: Sequence[int] = (1, 5, 10, 15),
) -> pd.DataFrame:
    if fixtures.empty:
        return pd.DataFrame()

    fx = _ensure_gw_col(fixtures)
    required = {"gw", "home_team", "away_team", "home_team_elo", "away_team_elo"}
    missing = [c for c in required if c not in fx.columns]
    if missing:
        raise ValueError(f"fixtures missing columns: {missing}")

    fx = fx.copy()
    fx["gw"] = fx["gw"].astype(int)

    home_rows = fx.assign(
        team_id=fx["home_team"],
        opp_id=fx["away_team"],
        is_home=1,
        team_elo=fx["home_team_elo"],
        opp_elo=fx["away_team_elo"],
    )
    away_rows = fx.assign(
        team_id=fx["away_team"],
        opp_id=fx["home_team"],
        is_home=0,
        team_elo=fx["away_team_elo"],
        opp_elo=fx["home_team_elo"],
    )
    team_fx = pd.concat([home_rows, away_rows], ignore_index=True)

    if teams_dim is not None and not teams_dim.empty:
        if "id" in teams_dim.columns and "strength" in teams_dim.columns:
            team_fx = team_fx.merge(
                teams_dim[["id", "strength"]].rename(
                    columns={"id": "opp_id", "strength": "opp_strength"}
                ),
                on="opp_id",
                how="left",
            )
        if "id" in teams_dim.columns and "elo" in teams_dim.columns:
            team_fx = team_fx.merge(
                teams_dim[["id", "elo"]].rename(
                    columns={"id": "team_id", "elo": "team_elo_dim"}
                ),
                on="team_id",
                how="left",
            )
            team_fx["team_elo"] = team_fx["team_elo"].fillna(team_fx["team_elo_dim"])
            team_fx = team_fx.drop(columns=["team_elo_dim"], errors="ignore")

    team_fx["elo_delta"] = team_fx["team_elo"] - team_fx["opp_elo"]

    min_gw = int(team_fx["gw"].min())
    max_gw = int(team_fx["gw"].max())
    teams = sorted(team_fx["team_id"].dropna().unique())

    rows = []
    for team_id in teams:
        team_rows = team_fx[team_fx["team_id"] == team_id]
        for current_gw in range(min_gw, max_gw + 1):
            row = {"team_id": team_id, "gw": current_gw}
            for h in horizons:
                upcoming = team_rows[
                    (team_rows["gw"] > current_gw) & (team_rows["gw"] <= current_gw + h)
                ]
                count = int(len(upcoming))
                row[f"fixture_count_next_{h}"] = count
                if count == 0:
                    row[f"home_share_next_{h}"] = pd.NA
                    row[f"opp_elo_avg_next_{h}"] = pd.NA
                    row[f"opp_strength_avg_next_{h}"] = pd.NA
                    row[f"elo_delta_avg_next_{h}"] = pd.NA
                else:
                    row[f"home_share_next_{h}"] = upcoming["is_home"].mean()
                    row[f"opp_elo_avg_next_{h}"] = upcoming["opp_elo"].mean()
                    row[f"opp_strength_avg_next_{h}"] = (
                        upcoming["opp_strength"].mean()
                        if "opp_strength" in upcoming.columns
                        else pd.NA
                    )
                    row[f"elo_delta_avg_next_{h}"] = upcoming["elo_delta"].mean()
            rows.append(row)

    return pd.DataFrame(rows)
