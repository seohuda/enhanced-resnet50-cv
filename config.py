import argparse
from typing import Any, Dict, Set

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def merge_config(
    args: argparse.Namespace,
    explicit_keys: Set[str],
    yaml_dict: Dict[str, Any],
) -> argparse.Namespace:
    merged = vars(args)

    for key, value in yaml_dict.items():
        dest = key.replace("-", "_")
        if dest not in merged:
            raise ValueError(f"Unknown config key: {key}")
        if dest not in explicit_keys:
            merged[dest] = value

    return argparse.Namespace(**merged)
