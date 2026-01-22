from __future__ import annotations

from pathlib import Path


def _slug(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def model_filename(position: str, target: str, horizon: int, ext: str = "joblib") -> str:
    suffix = ext.lstrip(".") if ext else "joblib"
    return f"{_slug(position)}__{_slug(target)}__h{int(horizon)}.{suffix}"


def model_path(
    output_dir: str | Path,
    position: str,
    target: str,
    horizon: int,
    ext: str = "joblib",
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out / model_filename(position, target, horizon, ext=ext)
