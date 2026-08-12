import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_paths() -> dict:
    with open(CONFIG_DIR / "paths.json") as file:
        paths = json.load(file)

    local_file = CONFIG_DIR / "paths.local.json"
    if local_file.exists():
        with open(local_file) as file:
            paths = _deep_merge(paths, json.load(file))

    return paths
