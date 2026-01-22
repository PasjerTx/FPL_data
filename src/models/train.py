from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
try:
    from joblib import dump  # type: ignore
    _MODEL_EXT = "joblib"
except Exception:  # pragma: no cover - fallback for environments without joblib
    import pickle

    _MODEL_EXT = "pkl"

    def dump(obj, filename):  # type: ignore
        with open(filename, "wb") as f:
            pickle.dump(obj, f)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.registry import model_path
try:
    from src.evaluation.plots import (
        plot_feature_importance,
        plot_pred_vs_actual,
        plot_residuals,
    )
except Exception:  # pragma: no cover
    plot_feature_importance = None
    plot_pred_vs_actual = None
    plot_residuals = None


DEFAULT_EXCLUDE_COLS = {
    "player_id",
    "gw",
    "team_id",
    "team_code",
    "code",
    "id",
}


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def numeric_like_columns(df: pd.DataFrame, threshold: float = 0.9) -> List[str]:
    cols: List[str] = []
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        non_null_ratio = float(converted.notna().mean())
        if non_null_ratio >= threshold:
            cols.append(col)
    return cols


def get_label_columns(targets: Sequence[str], horizons: Sequence[int]) -> List[str]:
    cols = []
    for t in targets:
        for h in horizons:
            cols.append(f"{t}_next_{h}_total")
            cols.append(f"{t}_next_{h}_per_week")
    return cols


def get_feature_columns(
    df: pd.DataFrame,
    targets: Sequence[str],
    horizons: Sequence[int],
    exclude_cols: Iterable[str] | None = None,
    include_cols: Iterable[str] | None = None,
) -> List[str]:
    numeric_cols = set(df.select_dtypes(include=["number", "bool"]).columns)
    numeric_like = set(numeric_like_columns(df))
    candidate = numeric_cols.union(numeric_like)

    label_cols = set(get_label_columns(targets, horizons))
    excluded = set(exclude_cols or []).union(DEFAULT_EXCLUDE_COLS).union(label_cols)

    if include_cols:
        include_set = {c for c in include_cols if c in df.columns}
        candidate = candidate.intersection(include_set)

    features = [c for c in df.columns if c in candidate and c not in excluded]
    return features


def _split_train_val(
    df: pd.DataFrame, holdout_gws: int, gw_col: str = "gw"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if holdout_gws <= 0 or gw_col not in df.columns:
        return df, pd.DataFrame()
    max_gw = int(df[gw_col].max())
    cutoff = max_gw - int(holdout_gws)
    train = df[df[gw_col] <= cutoff]
    val = df[df[gw_col] > cutoff]
    return train, val


def _prepare_xy(
    df: pd.DataFrame, feature_cols: Sequence[str], label_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    numeric_df = coerce_numeric(df)
    X = numeric_df[feature_cols].fillna(0)
    y = pd.to_numeric(numeric_df[label_col], errors="coerce")
    return X, y


def train_models(
    df: pd.DataFrame,
    positions: Sequence[str],
    targets: Sequence[str],
    horizons: Sequence[int],
    rf_params: Dict,
    output_dir: str | Path,
    holdout_gws: int = 3,
    exclude_cols: Iterable[str] | None = None,
    include_cols: Iterable[str] | None = None,
    position_targets: Dict[str, Sequence[str]] | None = None,
    plot_dir: str | Path | None = None,
) -> pd.DataFrame:
    if "position" not in df.columns:
        raise ValueError("Training dataset must include position column")

    # Build feature columns using union of all target labels to exclude
    all_targets = set(targets)
    if position_targets:
        for t_list in position_targets.values():
            all_targets.update(t_list)
    feature_cols = get_feature_columns(
        df=df,
        targets=sorted(all_targets),
        horizons=horizons,
        exclude_cols=exclude_cols,
        include_cols=include_cols,
    )

    metrics_rows: List[Dict] = []
    for position in positions:
        pos_df = df[df["position"] == position].copy()
        if pos_df.empty:
            continue
        pos_targets = (
            list(position_targets.get(position, targets))
            if position_targets
            else list(targets)
        )
        for target in pos_targets:
            for horizon in horizons:
                label_col = f"{target}_next_{horizon}_total"
                if label_col not in pos_df.columns:
                    continue
                subset = pos_df.dropna(subset=[label_col])
                if subset.empty:
                    continue

                train_df, val_df = _split_train_val(subset, holdout_gws=holdout_gws)
                X_train, y_train = _prepare_xy(train_df, feature_cols, label_col)
                X_val, y_val = _prepare_xy(val_df, feature_cols, label_col)

                model = RandomForestRegressor(**rf_params)
                model.fit(X_train, y_train)

                metrics = {
                    "position": position,
                    "target": target,
                    "horizon": int(horizon),
                    "train_size": len(train_df),
                    "val_size": len(val_df),
                }

                if len(val_df) > 0:
                    preds = model.predict(X_val)
                    metrics.update(
                        {
                            "r2": r2_score(y_val, preds),
                            "mae": mean_absolute_error(y_val, preds),
                            "rmse": mean_squared_error(y_val, preds, squared=False),
                        }
                    )
                    if plot_dir is not None and plot_pred_vs_actual and plot_residuals:
                        try:
                            plot_root = Path(plot_dir)
                            name = f"{position}__{target}__h{horizon}"
                            plot_pred_vs_actual(
                                y_val,
                                preds,
                                plot_root / "pred_vs_actual" / f"{name}.png",
                                title=f"{name} Pred vs Actual",
                            )
                            plot_residuals(
                                y_val,
                                preds,
                                plot_root / "residuals" / f"{name}.png",
                                title=f"{name} Residuals",
                            )
                        except Exception:
                            pass
                else:
                    metrics.update({"r2": None, "mae": None, "rmse": None})

                artifact = {
                    "model": model,
                    "feature_cols": feature_cols,
                    "label_col": label_col,
                    "position": position,
                    "target": target,
                    "horizon": int(horizon),
                }
                dump(artifact, model_path(output_dir, position, target, horizon, ext=_MODEL_EXT))

                if plot_dir is not None and plot_feature_importance:
                    try:
                        plot_feature_importance(
                            feature_cols,
                            model.feature_importances_,
                            Path(plot_dir)
                            / "feature_importance"
                            / f"{position}__{target}__h{horizon}.png",
                            title=f"{position} {target} h{horizon} Feature Importance",
                            top_n=30,
                        )
                    except Exception:
                        pass
                metrics_rows.append(metrics)

    return pd.DataFrame(metrics_rows)
