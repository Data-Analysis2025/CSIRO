"""Utilities for loading experiment configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    """Dataclass wrapper around a dictionary config."""

    data: Dict[str, Any]
    path: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data=data, path=path)

    def get(self, *keys: str, default: Any | None = None) -> Any:
        node: Any = self.data
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
        return node

    def ensure_dirs(self) -> None:
        """Ensure that runtime directories exist."""
        paths = self.data.get("paths", {})
        for key in ["processed_dir", "models_dir", "submissions_dir", "logs_dir"]:
            value = paths.get(key)
            if value is None:
                continue
            Path(value).mkdir(parents=True, exist_ok=True)


__all__ = ["Config"]
