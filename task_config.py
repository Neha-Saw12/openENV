"""Shared task configuration loaded from openenv.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parent
OPENENV_MANIFEST = ROOT / "openenv.yaml"


def load_openenv_manifest() -> Dict[str, Any]:
    """Load the environment manifest from disk."""
    with OPENENV_MANIFEST.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_task_configs() -> Dict[str, Dict[str, Any]]:
    """Return tasks keyed by task name."""
    manifest = load_openenv_manifest()
    tasks = manifest.get("tasks", [])
    return {task["name"]: task for task in tasks if "name" in task}


TASK_CONFIGS = load_task_configs()
DEFAULT_TASK_NAME = "smart_shop" if "smart_shop" in TASK_CONFIGS else next(
    iter(TASK_CONFIGS),
    "dynamic",
)


def get_task_config(task_name: str | None) -> Dict[str, Any] | None:
    """Fetch a task configuration by name."""
    if not task_name:
        return None
    return TASK_CONFIGS.get(task_name)
