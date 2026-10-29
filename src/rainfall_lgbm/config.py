"""Configuration loading utilities."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str | Path | None = None, override: dict | None = None) -> dict:
    """Load the base configuration, optionally merged with an override file/dict."""
    cfg = load_yaml(ROOT / "configs" / "base.yaml")
    if path is not None and Path(path).resolve() != (ROOT / "configs" / "base.yaml").resolve():
        cfg = deep_update(cfg, load_yaml(path))
    if override:
        cfg = deep_update(cfg, override)
    return cfg


def load_scenario(code: str) -> dict:
    sc = load_yaml(ROOT / "configs" / "scenarios" / f"{code}.yaml")
    if sc.get("code") != code:
        raise ValueError(f"Scenario file for {code} is inconsistent")
    return sc


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
