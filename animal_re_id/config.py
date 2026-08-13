import json
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_PATHS_FILE = PACKAGE_DIR / "data" / "paths.json"


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_path(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def load_paths(dataset: str, local_path: str | Path | None = None) -> dict:
    with open(DEFAULT_PATHS_FILE) as file:
        paths = json.load(file)

    if local_path is not None and Path(local_path).exists():
        with open(local_path) as file:
            paths = _deep_merge(paths, json.load(file))

    return paths[dataset]
