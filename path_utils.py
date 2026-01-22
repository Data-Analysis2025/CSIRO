"""Common helpers for resolving dataset paths."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

DEFAULT_DIRS: tuple[Path, ...] = (
    Path("data/csiro_biomass"),
    Path("data"),
)


def resolve_data_dir(preferred: str | Path | None = None, *, required_files: Iterable[str] | None = None) -> Path:
    """
    Return the first existing dataset directory.

    The function checks `preferred` first, then known defaults. A directory is
    considered valid when every file in `required_files` (defaults to train.csv)
    exists beneath it.  Raises FileNotFoundError if nothing matches.
    """
    candidates: list[Path] = []
    if preferred:
        candidates.append(Path(preferred))
    for extra in DEFAULT_DIRS:
        if not preferred or Path(preferred) != extra:
            candidates.append(extra)

    required = list(required_files or ["train.csv"])

    for candidate in candidates:
        if all((candidate / req).exists() for req in required):
            return candidate
    raise FileNotFoundError(
        f"Could not find dataset directory among {[str(c) for c in candidates]}; missing files {required}."
    )


__all__ = ["resolve_data_dir"]
