"""Shared helpers for the command-line scripts."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rainfall_lgbm.config import load_config  # noqa: E402


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None,
                   help="Optional YAML file overriding configs/base.yaml (e.g. tests/quick_config.yaml)")
    return p


def config_from(args) -> dict:
    return load_config(args.config)


def out_dir(cfg: dict, *parts) -> Path:
    from rainfall_lgbm.config import resolve
    p = resolve(cfg["paths"]["results"]).joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
