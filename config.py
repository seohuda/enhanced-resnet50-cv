import argparse
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def merge_config(
    args: argparse.Namespace,
    defaults: argparse.Namespace,
    yaml_dict: Dict[str, Any],
) -> argparse.Namespace:
    merged = vars(args)
    default_values = vars(defaults)

    for key, value in yaml_dict.items():
        dest = key.replace("-", "_")
        if dest not in merged:
            raise ValueError(f"Unknown config key: {key}")
        if merged[dest] == default_values.get(dest):
            merged[dest] = value

    return argparse.Namespace(**merged)
