from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd


REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "teams": ["code", "id", "name", "strength", "elo"],
    "players": ["player_id", "web_name", "team_code", "position"],
    "matches": [
        "gameweek",
        "home_team",
        "away_team",
        "home_team_elo",
        "away_team_elo",
        "finished",
        "match_id",
    ],
    "fixtures": [
        "gameweek",
        "home_team",
        "away_team",
        "home_team_elo",
        "away_team_elo",
        "finished",
        "match_id",
    ],
    "player_gameweek_stats": [
        "id",
        "gw",
        "minutes",
        "event_points",
        "expected_goals",
        "expected_assists",
        "defensive_contribution",
    ],
    "playerstats": [
        "id",
        "gw",
        "minutes",
        "event_points",
        "expected_goals",
        "expected_assists",
        "defensive_contribution",
    ],
    "playermatchstats": ["player_id", "match_id", "minutes_played"],
}


def validate_columns(df: pd.DataFrame, required: Iterable[str], table: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {table}: {missing}")


def validate_all(tables: Dict[str, pd.DataFrame]) -> None:
    for name, required in REQUIRED_COLUMNS.items():
        if name not in tables:
            raise ValueError(f"Missing table in load output: {name}")
        validate_columns(tables[name], required, name)


def list_required_columns(table: str) -> List[str]:
    if table not in REQUIRED_COLUMNS:
        raise KeyError(f"Unknown table: {table}")
    return REQUIRED_COLUMNS[table]
