from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def list_gw_dirs(by_gameweek_dir: str | Path) -> List[Tuple[str, int, Path]]:
    root = Path(by_gameweek_dir)
    if not root.exists():
        raise FileNotFoundError(f"By Gameweek directory not found: {root}")
    gw_dirs: List[Tuple[str, int, Path]] = []
    for p in root.iterdir():
        if p.is_dir() and p.name.lower().startswith("gw"):
            try:
                gw_num = int(p.name[2:])
            except ValueError:
                continue
            gw_dirs.append((p.name, gw_num, p))
    return sorted(gw_dirs, key=lambda x: x[1])


def load_csv_for_gw(
    gw_path: str | Path,
    filename: str,
    add_gw: bool = True,
    gw_num: int | None = None,
) -> pd.DataFrame:
    path = Path(gw_path) / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, low_memory=False)
    if add_gw and "gw" not in df.columns:
        if gw_num is None:
            name = Path(gw_path).name
            gw_num = int(name.replace("GW", ""))
        df.insert(0, "gw", gw_num)
    return df


def load_all_gws(
    by_gameweek_dir: str | Path,
    filename: str,
    add_gw: bool = True,
    allow_missing: bool = False,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    first_columns: List[str] | None = None
    for _, gw_num, gw_path in list_gw_dirs(by_gameweek_dir):
        path = gw_path / filename
        if not path.exists():
            if allow_missing:
                continue
            raise FileNotFoundError(f"Missing file: {path}")
        df = pd.read_csv(path, low_memory=False)
        if first_columns is None:
            first_columns = list(df.columns)
        if add_gw and "gw" not in df.columns:
            df.insert(0, "gw", gw_num)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        cols = first_columns if first_columns is not None else []
        return pd.DataFrame(columns=cols)
    return pd.concat(non_empty, ignore_index=True)


def load_tables(
    data_root: str | Path,
    season: str = "2025-2026",
    add_gw: bool = True,
) -> Dict[str, pd.DataFrame]:
    by_gw = Path(data_root) / season / "By Gameweek"
    return {
        "teams": load_all_gws(by_gw, "teams.csv", add_gw=add_gw),
        "players": load_all_gws(by_gw, "players.csv", add_gw=add_gw),
        "matches": load_all_gws(by_gw, "matches.csv", add_gw=add_gw),
        "fixtures": load_all_gws(by_gw, "fixtures.csv", add_gw=add_gw),
        "player_gameweek_stats": load_all_gws(
            by_gw, "player_gameweek_stats.csv", add_gw=add_gw
        ),
        "playerstats": load_all_gws(by_gw, "playerstats.csv", add_gw=add_gw),
        "playermatchstats": load_all_gws(by_gw, "playermatchstats.csv", add_gw=add_gw),
    }


def load_specific_tables(
    by_gameweek_dir: str | Path,
    filenames: Iterable[str],
    add_gw: bool = True,
    allow_missing: bool = False,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for name in filenames:
        out[name] = load_all_gws(
            by_gameweek_dir, name, add_gw=add_gw, allow_missing=allow_missing
        )
    return out
