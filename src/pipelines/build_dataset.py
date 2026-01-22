from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import pandas as pd

from src.features.availability_flags import add_substitution_rates, build_availability_flags
from src.features.fixture_features import build_team_fixture_difficulty
from src.features.match_features import build_playermatchstats_features
from src.features.player_features import add_player_rolling_features
from src.io.index import FinishedGwIndex, compute_finished_gws
from src.io.loaders import load_tables
from src.io.schema import validate_all
from src.labels.targets import add_horizon_labels, labelable_mask


@dataclass(frozen=True)
class BaseDataset:
    data: pd.DataFrame
    finished_index: FinishedGwIndex


@dataclass(frozen=True)
class TrainingDataset:
    data: pd.DataFrame
    finished_index: FinishedGwIndex


def _latest_by_key(df: pd.DataFrame, key: str, sort_col: str = "gw") -> pd.DataFrame:
    if df.empty:
        return df
    if sort_col not in df.columns:
        return df.drop_duplicates(subset=[key], keep="last").reset_index(drop=True)
    return (
        df.sort_values(sort_col)
        .drop_duplicates(subset=[key], keep="last")
        .reset_index(drop=True)
    )


def _build_team_gw_features(
    matches: pd.DataFrame, teams_dim: pd.DataFrame, max_finished_gw: int
) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame()

    matches = matches.copy()
    if "gw" not in matches.columns and "gameweek" in matches.columns:
        matches["gw"] = matches["gameweek"]
    matches = matches[matches["gw"].astype(int) <= int(max_finished_gw)]

    team_strength = teams_dim[["id", "strength"]].rename(
        columns={"id": "team_id", "strength": "team_strength"}
    )
    opp_strength = teams_dim[["id", "strength"]].rename(
        columns={"id": "opp_id", "strength": "opp_strength"}
    )

    home_rows = matches.assign(
        team_id=matches["home_team"],
        opp_id=matches["away_team"],
        is_home=1,
        team_elo=matches["home_team_elo"],
        opp_elo=matches["away_team_elo"],
    )
    away_rows = matches.assign(
        team_id=matches["away_team"],
        opp_id=matches["home_team"],
        is_home=0,
        team_elo=matches["away_team_elo"],
        opp_elo=matches["home_team_elo"],
    )

    tm = pd.concat([home_rows, away_rows], ignore_index=True)
    tm = tm.merge(team_strength, on="team_id", how="left")
    tm = tm.merge(opp_strength, on="opp_id", how="left")

    agg = (
        tm.groupby(["gw", "team_id"], as_index=False)
        .agg(
            fixture_count=("match_id", "count"),
            home_share=("is_home", "mean"),
            team_elo_avg=("team_elo", "mean"),
            opp_elo_avg=("opp_elo", "mean"),
            team_strength_avg=("team_strength", "mean"),
            opp_strength_avg=("opp_strength", "mean"),
        )
        .reset_index(drop=True)
    )
    return agg


def build_base_dataset(
    tables: Dict[str, pd.DataFrame],
    finished_index: FinishedGwIndex,
    min_minutes: int = 1,
) -> pd.DataFrame:
    pgw = tables["player_gameweek_stats"]
    if pgw.empty:
        pgw = tables["playerstats"]
    if pgw.empty:
        raise ValueError("player_gameweek_stats and playerstats are both empty")

    players = tables["players"]
    teams = tables["teams"]
    matches = tables["matches"]

    players_dim = _latest_by_key(players, key="player_id")
    players_dim = players_dim.drop(columns=["gw"], errors="ignore")
    teams_dim = _latest_by_key(teams, key="code")

    pgw = pgw.copy()
    if "gw" not in pgw.columns:
        raise ValueError("player_gameweek_stats missing gw column")
    pgw["gw"] = pgw["gw"].astype(int)
    pgw = pgw[pgw["gw"] <= finished_index.max_finished_gw]
    pgw = pgw[pgw["minutes"] >= int(min_minutes)]
    pgw = pgw.rename(columns={"id": "player_id"})

    players_dim = players_dim.merge(
        teams_dim[["code", "id", "name", "strength", "elo"]],
        left_on="team_code",
        right_on="code",
        how="left",
    ).rename(
        columns={
            "id": "team_id",
            "name": "team_name",
            "strength": "team_strength",
            "elo": "team_elo",
        }
    )

    team_gw = _build_team_gw_features(matches, teams_dim, finished_index.max_finished_gw)

    base = pgw.merge(players_dim, on="player_id", how="left")
    # Coalesce duplicated name columns from pgw and players_dim
    for col in ["first_name", "second_name", "web_name"]:
        col_x = f"{col}_x"
        col_y = f"{col}_y"
        if col in base.columns:
            continue
        if col_x in base.columns or col_y in base.columns:
            base[col] = base[col_x].combine_first(base[col_y])
            base = base.drop(columns=[c for c in [col_x, col_y] if c in base.columns])
    if not team_gw.empty:
        base = base.merge(team_gw, on=["gw", "team_id"], how="left")

    return base


def load_and_build_base(
    data_root: str | Path,
    season: str = "2025-2026",
    min_minutes: int = 1,
) -> BaseDataset:
    tables = load_tables(data_root=data_root, season=season, add_gw=True)
    validate_all(tables)
    finished_index = compute_finished_gws(tables["matches"])
    base = build_base_dataset(tables, finished_index, min_minutes=min_minutes)
    return BaseDataset(data=base, finished_index=finished_index)


def build_training_dataset(
    tables: Dict[str, pd.DataFrame],
    finished_index: FinishedGwIndex,
    rolling_windows: Sequence[int] = (3, 5, 10),
    horizons: Sequence[int] = (1, 5, 10, 15),
    min_minutes: int = 1,
    include_sub_rates: bool = True,
) -> pd.DataFrame:
    base = build_base_dataset(tables, finished_index, min_minutes=min_minutes)

    features = add_player_rolling_features(
        base,
        windows=rolling_windows,
        per90=True,
        include_current=True,
    )

    # Availability flags are computed on the full player_gameweek_stats table (including DNPs)
    flags_src = tables.get("player_gameweek_stats", pd.DataFrame())
    if flags_src.empty:
        flags_src = tables.get("playerstats", pd.DataFrame())
    if not flags_src.empty:
        flags = build_availability_flags(flags_src, window_gw=max(rolling_windows))
        features = features.merge(flags, on=["player_id", "gw"], how="left")

    if include_sub_rates:
        features = add_substitution_rates(
            features,
            playermatchstats=tables.get("playermatchstats", pd.DataFrame()),
            matches=tables.get("matches", pd.DataFrame()),
            window_gw=max(rolling_windows),
        )

    # Match-level stats aggregated to player-GW (adds extra detail beyond player_gameweek_stats)
    pm = tables.get("playermatchstats", pd.DataFrame())
    if not pm.empty:
        pm_features = build_playermatchstats_features(
            playermatchstats=pm,
            matches=tables.get("matches", pd.DataFrame()),
            agg=("sum", "mean"),
        )
        if not pm_features.empty:
            features = features.merge(pm_features, on=["player_id", "gw"], how="left")

    labeled = add_horizon_labels(features, horizons=horizons, require_all=False)
    mask = labelable_mask(labeled, finished_index.max_finished_gw, horizons=horizons)
    return labeled.loc[mask].reset_index(drop=True)


def build_feature_dataset(
    tables: Dict[str, pd.DataFrame],
    finished_index: FinishedGwIndex,
    rolling_windows: Sequence[int] = (3, 5, 10),
    min_minutes: int = 1,
    include_sub_rates: bool = True,
) -> pd.DataFrame:
    base = build_base_dataset(tables, finished_index, min_minutes=min_minutes)

    features = add_player_rolling_features(
        base,
        windows=rolling_windows,
        per90=True,
        include_current=True,
    )

    flags_src = tables.get("player_gameweek_stats", pd.DataFrame())
    if flags_src.empty:
        flags_src = tables.get("playerstats", pd.DataFrame())
    if not flags_src.empty:
        flags = build_availability_flags(flags_src, window_gw=max(rolling_windows))
        features = features.merge(flags, on=["player_id", "gw"], how="left")

    if include_sub_rates:
        features = add_substitution_rates(
            features,
            playermatchstats=tables.get("playermatchstats", pd.DataFrame()),
            matches=tables.get("matches", pd.DataFrame()),
            window_gw=max(rolling_windows),
        )

    # Match-level stats aggregated to player-GW (adds extra detail beyond player_gameweek_stats)
    pm = tables.get("playermatchstats", pd.DataFrame())
    if not pm.empty:
        pm_features = build_playermatchstats_features(
            playermatchstats=pm,
            matches=tables.get("matches", pd.DataFrame()),
            agg=("sum", "mean"),
        )
        if not pm_features.empty:
            features = features.merge(pm_features, on=["player_id", "gw"], how="left")

    # Upcoming fixture difficulty features (schedule-based, no result leakage)
    fixtures = tables.get("fixtures", pd.DataFrame())
    teams = tables.get("teams", pd.DataFrame())
    if not fixtures.empty:
        if "gw" not in teams.columns:
            teams = teams.copy()
        teams_dim = (
            teams.sort_values("gw").drop_duplicates(subset=["code"], keep="last")
            if "gw" in teams.columns
            else teams
        )
        fixture_diff = build_team_fixture_difficulty(
            fixtures=fixtures, teams_dim=teams_dim, horizons=(1, 5, 10, 15)
        )
        features = features.merge(fixture_diff, on=["team_id", "gw"], how="left")

    return features


def build_prediction_dataset(
    tables: Dict[str, pd.DataFrame],
    finished_index: FinishedGwIndex,
    rolling_windows: Sequence[int] = (3, 5, 10),
    min_minutes: int = 0,
    include_sub_rates: bool = True,
) -> pd.DataFrame:
    features = build_feature_dataset(
        tables,
        finished_index,
        rolling_windows=rolling_windows,
        min_minutes=min_minutes,
        include_sub_rates=include_sub_rates,
    )
    latest_gw = int(finished_index.max_finished_gw)
    return features[features["gw"] == latest_gw].reset_index(drop=True)


def load_and_build_training(
    data_root: str | Path,
    season: str = "2025-2026",
    rolling_windows: Sequence[int] = (3, 5, 10),
    horizons: Sequence[int] = (1, 5, 10, 15),
    min_minutes: int = 1,
    include_sub_rates: bool = True,
) -> TrainingDataset:
    tables = load_tables(data_root=data_root, season=season, add_gw=True)
    validate_all(tables)
    finished_index = compute_finished_gws(tables["matches"])
    data = build_training_dataset(
        tables,
        finished_index,
        rolling_windows=rolling_windows,
        horizons=horizons,
        min_minutes=min_minutes,
        include_sub_rates=include_sub_rates,
    )
    return TrainingDataset(data=data, finished_index=finished_index)


def _parse_int_list(value: str, default: Sequence[int]) -> Sequence[int]:
    if not value:
        return default
    try:
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid list value: {value}") from exc


def save_training_dataset(
    data_root: str | Path,
    output_path: str | Path,
    season: str = "2025-2026",
    rolling_windows: Sequence[int] = (3, 5, 10),
    horizons: Sequence[int] = (1, 5, 10, 15),
    min_minutes: int = 1,
    include_sub_rates: bool = True,
) -> Path:
    result = load_and_build_training(
        data_root=data_root,
        season=season,
        rolling_windows=rolling_windows,
        horizons=horizons,
        min_minutes=min_minutes,
        include_sub_rates=include_sub_rates,
    )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        result.data.to_parquet(out_path, index=False)
    else:
        result.data.to_csv(out_path, index=False)
    return out_path


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build and save training dataset")
    parser.add_argument("--data-root", default="data", help="Data root directory")
    parser.add_argument("--season", default="2025-2026", help="Season folder name")
    parser.add_argument(
        "--output",
        default="data/processed/training_dataset.parquet",
        help="Output path (.parquet or .csv)",
    )
    parser.add_argument(
        "--rolling-windows",
        default="3,5,10",
        help="Comma-separated rolling windows",
    )
    parser.add_argument(
        "--horizons",
        default="1,5,10,15",
        help="Comma-separated label horizons",
    )
    parser.add_argument(
        "--min-minutes",
        type=int,
        default=1,
        help="Minimum minutes to keep a row",
    )
    parser.add_argument(
        "--no-sub-rates",
        action="store_true",
        help="Disable substitution rate features",
    )
    args = parser.parse_args()

    windows = _parse_int_list(args.rolling_windows, (3, 5, 10))
    horizons = _parse_int_list(args.horizons, (1, 5, 10, 15))
    save_training_dataset(
        data_root=args.data_root,
        output_path=args.output,
        season=args.season,
        rolling_windows=windows,
        horizons=horizons,
        min_minutes=args.min_minutes,
        include_sub_rates=not args.no_sub_rates,
    )


if __name__ == "__main__":
    _main()
