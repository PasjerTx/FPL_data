from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from src.models.train import train_models


def _load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train per-position models")
    parser.add_argument(
        "--dataset",
        default="data/processed/training_dataset.parquet",
        help="Training dataset path (.parquet or .csv)",
    )
    parser.add_argument(
        "--config",
        default="configs/models.yaml",
        help="Model config path",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/models",
        help="Directory to save models",
    )
    parser.add_argument(
        "--metrics-out",
        default="reports/model_metrics.csv",
        help="Metrics output CSV",
    )
    parser.add_argument(
        "--plot-dir",
        default="reports/figures",
        help="Directory to save evaluation plots",
    )
    parser.add_argument(
        "--holdout-gws",
        type=int,
        default=None,
        help="Override holdout GWs (default from config)",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    df = _load_dataset(args.dataset)

    positions = config.get("positions", [])
    horizons = config.get("horizons", [])
    targets = config.get("targets", [])
    rf_params = config.get("rf", {})
    holdout = (
        args.holdout_gws
        if args.holdout_gws is not None
        else int(config.get("training", {}).get("final_holdout_gws", 3))
    )

    exclude_cols = config.get("feature_exclude")
    include_cols = config.get("feature_include")
    position_targets = config.get("position_targets")

    metrics = train_models(
        df=df,
        positions=positions,
        targets=targets,
        horizons=horizons,
        rf_params=rf_params,
        output_dir=args.output_dir,
        holdout_gws=holdout,
        exclude_cols=exclude_cols,
        include_cols=include_cols,
        position_targets=position_targets,
        plot_dir=args.plot_dir,
    )

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)


if __name__ == "__main__":
    main()
