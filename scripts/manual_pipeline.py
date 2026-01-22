from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd
import yaml

# Ensure project root is on sys.path when running directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.index import compute_finished_gws
from src.io.loaders import load_tables
from src.io.schema import validate_all
from src.models.predict import predict_from_models
from src.models.train import train_models
from src.pipelines.build_dataset import (
    build_prediction_dataset,
    load_and_build_training,
    save_training_dataset,
)
from src.reporting.export_predictions import export_position_predictions
from src.reporting.export_summaries import export_top_n_summaries


def _load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _parse_int_list(value: str, default: Sequence[int]) -> Sequence[int]:
    if not value:
        return default
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual end-to-end FPL pipeline")
    parser.add_argument("--data-root", default="data", help="Data root directory")
    parser.add_argument("--season", default="2025-2026", help="Season folder name")
    parser.add_argument(
        "--features-config",
        default="configs/features.yaml",
        help="Features config path",
    )
    parser.add_argument(
        "--models-config",
        default="configs/models.yaml",
        help="Models config path",
    )
    parser.add_argument(
        "--training-out",
        default="data/processed/training_dataset.parquet",
        help="Training dataset output path",
    )
    parser.add_argument(
        "--models-dir",
        default="outputs/models",
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--metrics-out",
        default="reports/model_metrics.csv",
        help="Metrics output CSV",
    )
    parser.add_argument(
        "--full-output",
        default="outputs/predictions/all_predictions.csv",
        help="Full predictions output path",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/predictions",
        help="Output directory for prediction CSVs",
    )
    parser.add_argument(
        "--summary-dir",
        default="outputs/predictions/summaries",
        help="Summary output directory",
    )
    parser.add_argument(
        "--summary-top-n",
        type=int,
        default=20,
        help="Top N players per position for summary",
    )
    parser.add_argument(
        "--summary-target",
        default="points",
        help="Target to sort summaries by (e.g., points)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only run prediction steps",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip prediction steps",
    )
    parser.add_argument(
        "--rolling-windows",
        default="",
        help="Override rolling windows (comma-separated)",
    )
    parser.add_argument(
        "--horizons",
        default="",
        help="Override horizons (comma-separated)",
    )
    parser.add_argument(
        "--min-minutes-train",
        type=int,
        default=1,
        help="Minimum minutes to keep a row for training",
    )
    parser.add_argument(
        "--min-minutes-predict",
        type=int,
        default=0,
        help="Minimum minutes to keep a row for prediction (0 includes DNPs)",
    )
    parser.add_argument(
        "--no-sub-rates",
        action="store_true",
        help="Disable substitution rate features",
    )
    parser.add_argument(
        "--plot-dir",
        default="reports/figures",
        help="Directory to save evaluation plots",
    )
    args = parser.parse_args()

    features_cfg = _load_config(args.features_config)
    models_cfg = _load_config(args.models_config)

    rolling_windows = _parse_int_list(
        args.rolling_windows, features_cfg.get("rolling_windows", [3, 5, 10])
    )
    horizons = _parse_int_list(args.horizons, models_cfg.get("horizons", [1, 5, 10, 15]))

    # 1) Build and save training dataset
    training_path = Path(args.training_out)
    if not args.skip_train:
        save_training_dataset(
            data_root=args.data_root,
            output_path=training_path,
            season=args.season,
            rolling_windows=rolling_windows,
            horizons=horizons,
            min_minutes=args.min_minutes_train,
            include_sub_rates=not args.no_sub_rates,
        )

        # 2) Train models
        df = _load_dataset(training_path)
        metrics = train_models(
            df=df,
            positions=models_cfg.get("positions", []),
            targets=models_cfg.get("targets", []),
            horizons=horizons,
            rf_params=models_cfg.get("rf", {}),
            output_dir=args.models_dir,
            holdout_gws=int(models_cfg.get("training", {}).get("final_holdout_gws", 3)),
            exclude_cols=models_cfg.get("feature_exclude"),
            include_cols=models_cfg.get("feature_include"),
            position_targets=models_cfg.get("position_targets"),
            plot_dir=args.plot_dir,
        )
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(metrics_path, index=False)

    # 3) Predict next GW and export
    if not args.skip_predict:
        tables = load_tables(data_root=args.data_root, season=args.season, add_gw=True)
        validate_all(tables)
        finished_index = compute_finished_gws(tables["matches"])
        features = build_prediction_dataset(
            tables=tables,
            finished_index=finished_index,
            rolling_windows=rolling_windows,
            min_minutes=args.min_minutes_predict,
            include_sub_rates=not args.no_sub_rates,
        )
        predictions = predict_from_models(args.models_dir, features)
        pred_path = Path(args.full_output)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(pred_path, index=False)

        # Export using union of global targets and any position-specific targets
        export_targets = list(models_cfg.get("targets", []))
        pos_targets = models_cfg.get("position_targets") or {}
        for t_list in pos_targets.values():
            for t in t_list:
                if t not in export_targets:
                    export_targets.append(t)

        export_position_predictions(
            predictions=predictions,
            output_dir=args.output_dir,
            targets=export_targets,
            horizons=horizons,
        )
        export_top_n_summaries(
            predictions=predictions,
            output_dir=args.summary_dir,
            target=args.summary_target,
            horizons=horizons,
            top_n=args.summary_top_n,
            sort_horizon=10,
        )


if __name__ == "__main__":
    main()
