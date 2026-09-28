from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_profile(profiles: Mapping[str, Any], name: str, seen: tuple[str, ...] = ()) -> dict[str, Any]:
    if name in seen:
        chain = " -> ".join((*seen, name))
        raise ValueError(f"Recursive profile inheritance: {chain}")
    if name not in profiles:
        available = ", ".join(sorted(str(key) for key in profiles))
        raise KeyError(f"Unknown config profile '{name}'. Available profiles: {available}")

    profile = copy.deepcopy(profiles[name] or {})
    parent = profile.pop("profile_extends", None)
    if not parent:
        return profile
    base = _resolve_profile(profiles, str(parent), (*seen, name))
    return deep_update(base, profile)


def load_config(path: str | os.PathLike[str], profile: str | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    parent = config.get("extends")
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent.parent / parent_path
            if not parent_path.exists():
                parent_path = config_path.parent / parent
        base = load_config(parent_path)
        config = deep_update(base, config)

    profiles = config.get("profiles") or {}
    selected_profile = profile or os.environ.get("HIFEN_PROFILE") or config.get("default_profile")
    if profiles:
        if not selected_profile:
            available = ", ".join(sorted(str(key) for key in profiles))
            raise ValueError(f"Config {config_path} defines profiles but no profile was selected. Available profiles: {available}")
        profile_config = _resolve_profile(profiles, str(selected_profile))
        config = deep_update(config, profile_config)
        config["_profile"] = str(selected_profile)
    elif selected_profile:
        config["_profile"] = str(selected_profile)

    config.pop("profiles", None)
    config.pop("default_profile", None)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config


def ensure_dirs(config: Mapping[str, Any], include_runtime: bool = True) -> None:
    storage = config.get("storage", {}) or {}
    keys = [
        "project_artifact_dir",
        "manifest_dir",
        "graph_dir",
        "checkpoint_dir",
        "label_map_dir",
        "split_dir",
        "metrics_dir",
        "reproducibility_dir",
    ]
    for key in keys:
        path = storage.get(key)
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)
    if include_runtime:
        runtime = config.get("runtime_outputs", {}) or {}
        for key in ("log_dir", "inference_result_dir"):
            path = runtime.get(key)
            if path:
                Path(path).mkdir(parents=True, exist_ok=True)


def dataset_resource_dirs(config: Mapping[str, Any], dataset_name: str) -> dict[str, Path]:
    save_dir = Path(config["storage"]["save_dir"])
    spec = (config.get("datasets", {}) or {})[dataset_name]
    output_subdir = spec.get("output_subdir", dataset_name)
    root = save_dir / output_subdir
    return {
        "crystal": root / "crystal",
        "predicted": root / "predicted",
        "complete": root / "complete",
        "embedding": root / "embedding",
    }
