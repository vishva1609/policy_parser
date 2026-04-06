"""
Load centralised pipeline settings from config.yaml.

Mirrors the config file pattern used in archit1012/qa-bot-llm (root config.yaml).
"""
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config.yaml"


def load_pipeline_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse config.yaml into a dict. Returns {} if the file is missing or empty.

    Args:
        path: Optional path to YAML. Defaults to repo-root config.yaml.
    """
    p = Path(path) if path else _default_config_path()
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}
