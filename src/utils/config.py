"""Загрузка YAML-конфигов экспериментов."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Корень репозитория (…/orient), независимо от cwd."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return ((base or project_root()) / p).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_experiment_config(path: str | Path, root: Path | None = None) -> dict[str, Any]:
    """
    Загружает конфиг эксперимента и встраивает classes_config.
    Все относительные пути резолвятся от корня репозитория (не от cwd) —
    удобно для Jupyter и облачных серверов.
    """
    root = Path(root) if root is not None else project_root()
    cfg_path = resolve_path(path, root)
    cfg = load_yaml(cfg_path)
    cfg["_project_root"] = str(root)
    cfg["_config_path"] = str(cfg_path)

    data = cfg.setdefault("data", {})
    classes_path = data.get("classes_config")
    if classes_path:
        classes_cfg = load_yaml(resolve_path(classes_path, root))
        cfg["classes"] = {int(k): v for k, v in classes_cfg["classes"].items()}
        cfg["label_priority"] = list(classes_cfg["label_priority"])
        cfg["empty_classes"] = list(classes_cfg.get("empty_classes", []))
        cfg["critical_classes"] = list(classes_cfg.get("critical_classes", []))
        cfg["linear_classes"] = list(classes_cfg.get("linear_classes", []))
        data["classes_config"] = str(resolve_path(classes_path, root))

    for key in ("dataset_root", "index_csv", "class_balance_json"):
        if key in data and data[key]:
            data[key] = str(resolve_path(data[key], root))

    train = cfg.setdefault("train", {})
    for key in ("checkpoint_dir", "runs_dir"):
        if key in train and train[key]:
            train[key] = str(resolve_path(train[key], root))
    if train.get("resume"):
        train["resume"] = str(resolve_path(train["resume"], root))
    if train.get("finetune_from"):
        train["finetune_from"] = str(resolve_path(train["finetune_from"], root))

    return cfg


def deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out
