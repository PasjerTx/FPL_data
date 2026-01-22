from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd
import yaml

from src.io.index import compute_finished_gws
from src.io.loaders import load_tables
from src.io.schema import validate_all
from src.models.predict import predict_from_models
from src.pipelines.build_dataset import build_prediction_dataset
from src.reporting.export_predictions import export_position_predictions
from src.reporting.export_summaries import export_top_n_summaries


def _load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_int_list(value: str, default: Sequence[int]) -> Sequence[int]:
    if not value:
        return default
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate next-GW predictions")
    parser.add_argument("--data-root", default="data", help="Data root directory")
    parser.add_argument("--season", default="2025-2026", help="Season folder name")
    parser.add_argument("--models-dir", default="outputs/models", help="Model artifacts dir")
    parser.add_argument(
        "--output-dir",
        default="outputs/predictions",
        help="Output directory for prediction CSVs",
    )
    parser.add_argument(
        "--config",
        default="configs/models.yaml",
        help="Model config path (for targets/horizons)",
    )
    parser.add_argument(
        "--rolling-windows",
        default="3,5,10",
        help="Comma-separated rolling windows",
    )
    parser.add_argument(
        "--min-minutes",
        type=int,
        default=0,
        help="Minimum minutes to keep a row (0 includes DNPs for prediction)",
    )
    parser.add_argument(
        "--no-sub-rates",
        action="store_true",
        help="Disable substitution rate features",
    )
    parser.add_argument(
        "--full-output",
        default="outputs/predictions/all_predictions.csv",
        help="Path to save full predictions table",
    )
    parser.add_argument(
        "--summary-dir",
        default="outputs/predictions/summaries",
        help="Directory to save top-N summary CSVs",
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
        "--no-summary",
        action="store_true",
        help="Disable summary CSV generation",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    targets = config.get("targets", [])
    horizons = config.get("horizons", [])
    rolling_windows = _parse_int_list(args.rolling_windows, (3, 5, 10))

    tables = load_tables(data_root=args.data_root, season=args.season, add_gw=True)
    validate_all(tables)
    finished_index = compute_finished_gws(tables["matches"])
    features = build_prediction_dataset(
        tables,
        finished_index,
        rolling_windows=rolling_windows,
        min_minutes=args.min_minutes,
        include_sub_rates=not args.no_sub_rates,
    )

    predictions = predict_from_models(args.models_dir, features)
    pred_cols = [c for c in predictions.columns if c.endswith("_pred")]
    if pred_cols:
        predictions[pred_cols] = (
            predictions[pred_cols].apply(pd.to_numeric, errors="coerce").round(1)
        )
    full_path = Path(args.full_output)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(full_path, index=False)

    # Export using union of global targets and any position-specific targets
    export_targets = list(targets)
    pos_targets = config.get("position_targets") or {}
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

    if not args.no_summary:
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
