from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

try:
    from joblib import load as _joblib_load  # type: ignore
except Exception:  # pragma: no cover
    _joblib_load = None
import pickle

_LOAD_EXTS = (".joblib", ".pkl")


def _load_any(path: Path):
    if path.suffix.lower() == ".joblib" and _joblib_load is not None:
        return _joblib_load(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def _list_model_files(models_dir: str | Path) -> List[Path]:
    root = Path(models_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    files: List[Path] = []
    for ext in _LOAD_EXTS:
        files.extend(root.glob(f"*{ext}"))
    return sorted(files)


def predict_from_models(
    models_dir: str | Path,
    features: pd.DataFrame,
    id_cols: Iterable[str] = (
        "player_id",
        "gw",
        "first_name",
        "second_name",
        "web_name",
        "team_name",
        "position",
        "now_cost",
    ),
) -> pd.DataFrame:
    model_files = _list_model_files(models_dir)
    if not model_files:
        raise FileNotFoundError(f"No model artifacts found in {models_dir}")

    base_cols = [c for c in id_cols if c in features.columns]
    key_cols = [c for c in ["player_id", "gw"] if c in base_cols]
    if not key_cols:
        key_cols = base_cols
    out = features[base_cols].copy()
    out = out.loc[:, ~out.columns.duplicated()]
    out = out.set_index(key_cols)

    for model_path in model_files:
        artifact = _load_any(model_path)
        model = artifact.get("model")
        feature_cols = artifact.get("feature_cols", [])
        position = artifact.get("position")
        target = artifact.get("target")
        horizon = artifact.get("horizon")
        if model is None or not feature_cols:
            continue

        df = features
        if position and "position" in df.columns:
            df = df[df["position"] == position]
        if df.empty:
            continue

        # Ensure all feature columns exist and are numeric
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        preds = model.predict(X)
        pred_col = f"{target}_next_{horizon}_total_pred"

        pred_df = df[[c for c in base_cols if c in df.columns]].copy()
        pred_df[pred_col] = preds
        pred_df = pred_df.reset_index(drop=True)

        pred_df = pred_df.loc[:, ~pred_df.columns.duplicated()]
        pred_df = pred_df.set_index(key_cols)

        if pred_col not in out.columns:
            out[pred_col] = pred_df[pred_col]
        else:
            out[pred_col] = out[pred_col].fillna(pred_df[pred_col])

    # Per-week convenience columns
    out = out.reset_index()

    for col in list(out.columns):
        if col.endswith("_total_pred") and "_next_" in col:
            try:
                h = int(col.split("_next_")[1].split("_")[0])
            except Exception:
                continue
            out[col.replace("_total_pred", "_per_week_pred")] = out[col] / float(h)

    return out
