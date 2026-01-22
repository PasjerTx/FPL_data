from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


def _mpl():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from exc
    return plt


def plot_pred_vs_actual(y_true, y_pred, path: str | Path, title: str) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=10)
    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(y_true, y_pred, path: str | Path, title: str) -> None:
    plt = _mpl()
    residuals = [p - t for p, t in zip(y_pred, y_true)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=40, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Residual (Pred - Actual)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(
    feature_cols: Sequence[str],
    importances: Sequence[float],
    path: str | Path,
    title: str,
    top_n: int = 30,
) -> None:
    plt = _mpl()
    pairs = list(zip(feature_cols, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:top_n]
    labels = [p[0] for p in top][::-1]
    values = [p[1] for p in top][::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(labels, values)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
