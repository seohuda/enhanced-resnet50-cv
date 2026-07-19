import argparse

import pytest

from config import load_config, merge_config


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="se_resnet50")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--cutmix", action="store_true")
    return parser


def test_load_config_reads_yaml(tmp_path):
    config_path = tmp_path / "test.yaml"
    config_path.write_text("model: resnet50\nepochs: 50\n")
    result = load_config(str(config_path))
    assert result == {"model": "resnet50", "epochs": 50}


def test_merge_config_yaml_fills_unset_cli_args():
    parser = make_parser()
    args = parser.parse_args([])
    defaults = parser.parse_args([])
    merged = merge_config(args, defaults, {"model": "resnet50", "epochs": 50})
    assert merged.model == "resnet50"
    assert merged.epochs == 50


def test_merge_config_cli_overrides_yaml():
    parser = make_parser()
    args = parser.parse_args(["--model", "se_resnet50"])
    defaults = parser.parse_args([])
    merged = merge_config(args, defaults, {"model": "resnet50", "epochs": 50})
    assert merged.model == "se_resnet50"
    assert merged.epochs == 50


def test_merge_config_applies_yaml_store_true_flag():
    parser = make_parser()
    args = parser.parse_args([])
    defaults = parser.parse_args([])
    merged = merge_config(args, defaults, {"cutmix": True})
    assert merged.cutmix is True


def test_merge_config_rejects_unknown_key():
    parser = make_parser()
    args = parser.parse_args([])
    defaults = parser.parse_args([])
    with pytest.raises(ValueError):
        merge_config(args, defaults, {"nonexistent_flag": 1})
