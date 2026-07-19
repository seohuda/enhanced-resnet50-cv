# Portfolio Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden `enhanced-resnet50-cv` for portfolio/hiring-demo use: real pytest coverage + CI, YAML config management, three well-chosen training technique additions (MixUp, label smoothing, EMA), and lightweight performance/export polish (ONNX export, batch inference, `torch.compile`).

**Architecture:** All changes are additive and opt-in. Existing CLI flags, `run_experiments.sh`, and the 4 existing ablation configs must keep behaving exactly as before unless a user opts into a new flag. New logic (MixUp selection, EMA, config merge) is factored into small, independently unit-testable functions rather than buried inline in `main()`, so each can be tested without running real training.

**Tech Stack:** Python 3.10, PyTorch 2.x, pytest, ruff, PyYAML, ONNX/ONNXRuntime, GitHub Actions.

## Global Constraints

- Existing runtime deps stay pinned as-is: `torch>=2.0.0`, `torchvision>=0.15.0`, `numpy>=1.24.0`, `matplotlib>=3.7.0`, `Pillow>=9.4.0`.
- New runtime dep: `PyYAML>=6.0` (added to `requirements.txt` — needed by `train.py` at runtime for `--config`).
- New dev-only deps (added to `requirements-dev.txt`, not `requirements.txt`): `pytest>=7.4.0`, `pytest-cov>=4.1.0`, `ruff>=0.4.0`, `onnx>=1.15.0`, `onnxruntime>=1.17.0`.
- CI runs on Python 3.10, ubuntu-latest, CPU only — no CUDA runner.
- Every new CLI flag defaults to today's behavior (off/0/None) — no existing config or script changes behavior silently.
- No placeholders, no TODOs — every task below ships working, tested code.
- Follow existing code style: no type-hint-only refactors change behavior; `Bottleneck`/`ResNet50`/etc. class structure in `model.py` is not restructured.

---

### Task 1: Convert test suite to real pytest

**Files:**
- Create: `tests/conftest.py`
- Modify: `tests/test_model.py` (full rewrite)
- Create: `tests/test_utils.py`
- Create: `requirements-dev.txt`
- Create: `pytest.ini`

**Interfaces:**
- Consumes: `build_model` from `model.py` (existing); `cutmix_data`, `set_seed` from `utils.py` (existing).
- Produces: pytest-discoverable suite under `tests/`. `tests/conftest.py` puts the repo root on `sys.path` so no test file needs a manual `sys.path.insert` hack. All later tasks add tests to `tests/test_utils.py` or new `tests/test_*.py` files that rely on this same `conftest.py`.

- [ ] **Step 1: Create `tests/conftest.py`**

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

- [ ] **Step 2: Rewrite `tests/test_model.py`**

Replace the entire file contents with:

```python
import pytest
import torch

from model import build_model


@pytest.mark.parametrize("model_name", ["resnet50", "se_resnet50"])
def test_model_output_shape(model_name):
    model = build_model(model_name, num_classes=100)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)


def test_se_resnet50_has_more_params_than_baseline():
    baseline = build_model("resnet50", num_classes=100)
    se_model = build_model("se_resnet50", num_classes=100)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    se_params = sum(p.numel() for p in se_model.parameters())
    assert se_params > baseline_params


def test_model_invalid_name_raises_value_error():
    with pytest.raises(ValueError):
        build_model("invalid_model", num_classes=100)
```

- [ ] **Step 3: Create `tests/test_utils.py`**

```python
import os

import torch

from model import build_model
from utils import cutmix_data, set_seed


def test_cutmix_preserves_shape():
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    x_mixed, y_a, y_b, lam = cutmix_data(x.clone(), y.clone(), alpha=1.0)
    assert x_mixed.shape == (4, 3, 32, 32)
    assert y_a.shape == (4,)
    assert y_b.shape == (4,)


def test_cutmix_lambda_range():
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 100, (8,))
    for _ in range(50):
        _, _, _, lam = cutmix_data(x.clone(), y.clone(), alpha=1.0)
        assert 0.0 <= lam <= 1.0


def test_seed_reproducibility():
    set_seed(42)
    model1 = build_model("resnet50", num_classes=100)
    x = torch.randn(1, 3, 32, 32)
    out1 = model1(x)

    set_seed(42)
    model2 = build_model("resnet50", num_classes=100)
    out2 = model2(x)

    assert torch.allclose(out1, out2)


def test_checkpoint_save_load(tmp_path):
    model = build_model("se_resnet50", num_classes=100)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    state = {
        "epoch": 10,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": 75.5,
        "config": {"model": "se_resnet50", "epochs": 200},
    }

    ckpt_path = os.path.join(str(tmp_path), "test_ckpt.pth")
    torch.save(state, ckpt_path)

    loaded = torch.load(ckpt_path, map_location="cpu")
    assert loaded["epoch"] == 10
    assert loaded["best_val_acc"] == 75.5
    assert loaded["config"]["model"] == "se_resnet50"

    model2 = build_model("se_resnet50", num_classes=100)
    model2.load_state_dict(loaded["model_state_dict"])

    model.eval()
    model2.eval()
    x = torch.randn(1, 3, 32, 32)
    assert torch.allclose(model(x), model2(x))
```

- [ ] **Step 4: Create `requirements-dev.txt`**

```
-r requirements.txt
pytest>=7.4.0
pytest-cov>=4.1.0
ruff>=0.4.0
```

- [ ] **Step 5: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
```

- [ ] **Step 6: Install dev deps and run the suite**

```bash
pip install -r requirements-dev.txt
pytest -v
```

Expected: all 6 tests pass (`test_model_output_shape[resnet50]`,
`test_model_output_shape[se_resnet50]`,
`test_se_resnet50_has_more_params_than_baseline`,
`test_model_invalid_name_raises_value_error`,
`test_cutmix_preserves_shape`, `test_cutmix_lambda_range`,
`test_seed_reproducibility`, `test_checkpoint_save_load` — 8 total).

- [ ] **Step 7: Commit**

```bash
git add tests/conftest.py tests/test_model.py tests/test_utils.py requirements-dev.txt pytest.ini
git commit -m "test: migrate test suite to pytest"
```

---

### Task 2: Add CI workflow and ruff lint config

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `ruff.toml`

**Interfaces:**
- Consumes: `requirements.txt`, `requirements-dev.txt` (Task 1), `pytest.ini` (Task 1).
- Produces: a GitHub Actions workflow named `CI` that later tasks' tests automatically run under once pushed.

- [ ] **Step 1: Create `ruff.toml`**

```toml
line-length = 120
target-version = "py310"
exclude = ["data", "runs", "results", "*.egg-info"]

[lint]
select = ["E", "F", "W", "I"]
ignore = ["E501"]
```

- [ ] **Step 2: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt -r requirements-dev.txt
      - name: Lint with ruff
        run: ruff check .
      - name: Run tests
        run: pytest --cov=. --cov-report=term-missing
```

- [ ] **Step 3: Run the same checks locally to confirm they pass before relying on CI**

```bash
ruff check .
pytest --cov=. --cov-report=term-missing
```

Expected: `ruff check .` reports no errors; `pytest` shows all tests passing with a
coverage summary printed.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml ruff.toml
git commit -m "ci: add GitHub Actions workflow and ruff config"
```

---

### Task 3: Add type hints across existing modules

**Files:**
- Modify: `model.py`
- Modify: `utils.py`
- Modify: `train.py`
- Modify: `inference.py`
- Create: `mypy.ini`

**Interfaces:**
- Consumes: nothing new — pure annotation addition, no behavior change.
- Produces: nothing later tasks depend on structurally, but later tasks' new functions should follow the same annotation style established here.

- [ ] **Step 1: Add type hints to `model.py`**

Update each signature (bodies unchanged):

```python
from typing import Optional

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        use_se: bool = False,
        reduction: int = 16,
    ) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class ResNet50(nn.Module):
    def __init__(self, num_classes: int = 100, use_se: bool = False, reduction: int = 16) -> None:
        ...

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        ...

    def _initialize_weights(self) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


def build_model(model_name: str = "se_resnet50", num_classes: int = 100) -> nn.Module:
    ...


def build_se_resnet50(num_classes: int = 100) -> nn.Module:
    ...
```

Only the signatures change; leave every method body exactly as-is.

- [ ] **Step 2: Add type hints to `utils.py`**

```python
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np


def cutmix_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    ...


def rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    ...


class SoftLabelCrossEntropyLoss(nn.Module):
    def __init__(self) -> None:
        ...

    def forward(
        self,
        outputs: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        ...


def set_seed(seed: int) -> None:
    ...
```

- [ ] **Step 3: Add type hints to `train.py`**

```python
import argparse
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    ...


def get_dataloaders(
    args: argparse.Namespace, device: torch.device
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ...


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, float]:
    ...


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, amp_enabled: bool = False
) -> Tuple[float, float, float]:
    ...


def save_checkpoint(state: dict, filepath: str) -> None:
    ...


def main() -> None:
    ...
```

- [ ] **Step 4: Add type hints to `inference.py`**

```python
from typing import List, Tuple

import torch
import torch.nn as nn


def load_model(
    checkpoint_path: str, device: torch.device, model_name: str = "se_resnet50"
) -> nn.Module:
    ...


def preprocess_image(image_path: str) -> torch.Tensor:
    ...


def predict(
    model: nn.Module, image_tensor: torch.Tensor, device: torch.device, top_k: int = 5
) -> List[Tuple[str, float]]:
    ...


def main() -> None:
    ...
```

- [ ] **Step 5: Create `mypy.ini`**

```ini
[mypy]
python_version = 3.10
ignore_missing_imports = True
check_untyped_defs = True
```

- [ ] **Step 6: Verify nothing broke**

```bash
pytest -v
mypy . || true
```

Expected: `pytest` still shows all tests passing (annotations don't change
runtime behavior). `mypy .` is informational only — do not fix its findings
as part of this task; it's a local tool, not a CI gate.

- [ ] **Step 7: Commit**

```bash
git add model.py utils.py train.py inference.py mypy.ini
git commit -m "refactor: add type hints across model, utils, train, inference"
```

---

### Task 4: Replace `print()` with structured logging in `train.py`

**Files:**
- Modify: `train.py`
- Create: `tests/test_train.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `setup_logging(output_dir: str) -> logging.Logger`, importable from `train.py`. Later tasks (5, 6, 7, 8, 9) add more functions to `train.py` and more tests to `tests/test_train.py`, so this file/module now exists for them to extend.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train.py`:

```python
import os

from train import setup_logging


def test_setup_logging_creates_file_and_writes_message(tmp_path):
    logger = setup_logging(str(tmp_path))
    logger.info("hello world")
    for handler in logger.handlers:
        handler.flush()

    log_path = os.path.join(str(tmp_path), "train.log")
    assert os.path.isfile(log_path)
    with open(log_path) as f:
        content = f.read()
    assert "hello world" in content


def test_setup_logging_does_not_duplicate_handlers_on_repeat_calls(tmp_path):
    logger1 = setup_logging(str(tmp_path))
    handler_count_after_first = len(logger1.handlers)

    logger2 = setup_logging(str(tmp_path))
    handler_count_after_second = len(logger2.handlers)

    assert handler_count_after_first == handler_count_after_second
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_train.py -v
```

Expected: FAIL with `ImportError: cannot import name 'setup_logging' from 'train'`.

- [ ] **Step 3: Add `setup_logging` to `train.py` and wire it into `main()`**

Add near the top of `train.py` (after the existing imports, before `parse_args`):

```python
import logging


def setup_logging(output_dir: str) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
```

Then in `main()`, replace the existing block:

```python
    os.makedirs(args.output_dir, exist_ok=True)

    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model} | CutMix: {args.cutmix} | Seed: {args.seed}")

    train_loader, val_loader, test_loader = get_dataloaders(args, device)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = build_model(args.model, num_classes=100).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
```

with:

```python
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)

    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model} | CutMix: {args.cutmix} | Seed: {args.seed}")

    train_loader, val_loader, test_loader = get_dataloaders(args, device)
    logger.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = build_model(args.model, num_classes=100).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {num_params:,}")
```

Then replace every remaining `print(...)` call later in `main()` (the
resume message, the epoch table header/rows, the final best/test accuracy
lines, and the "Results saved to" line) with the equivalent `logger.info(...)`
call, keeping the exact same message text/formatting — only the call
changes from `print` to `logger.info`.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_train.py -v
```

Expected: PASS for both tests.

- [ ] **Step 5: Run the full suite to make sure nothing else broke**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add train.py tests/test_train.py
git commit -m "feat: replace print() with structured logging in train.py"
```

---

### Task 5: Add YAML config management

**Files:**
- Create: `config.py`
- Create: `configs/baseline.yaml`
- Create: `configs/cutmix.yaml`
- Create: `configs/se.yaml`
- Create: `configs/se_cutmix.yaml`
- Modify: `train.py`
- Modify: `requirements.txt`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `load_config(path: str) -> dict`, `merge_config(args: argparse.Namespace, defaults: argparse.Namespace, yaml_dict: dict) -> argparse.Namespace` in `config.py`. `train.py`'s `parse_args()` keeps its existing signature/return type (`argparse.Namespace`) — no other task depends on `config.py` internals.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'config'` (or
`ImportError`).

- [ ] **Step 3: Create `config.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: PASS for all 5 tests.

- [ ] **Step 5: Add `--config` support to `train.py`**

Rename the existing `parse_args()` body into a `build_parser()` helper and
add `--config` plus the merge step. Replace:

```python
def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet50 variants on CIFAR-100")
    parser.add_argument("--model", type=str, default="se_resnet50",
                        choices=["resnet50", "se_resnet50"],
                        help="Model architecture")
```

with:

```python
from config import load_config, merge_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ResNet50 variants on CIFAR-100")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file (CLI flags override YAML values)")
    parser.add_argument("--model", type=str, default="se_resnet50",
                        choices=["resnet50", "se_resnet50"],
                        help="Model architecture")
```

Leave every other `parser.add_argument(...)` call in that function body
exactly as-is. Then change the function's closing lines from:

```python
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()
```

to:

```python
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    if args.config:
        defaults = parser.parse_args([])
        yaml_dict = load_config(args.config)
        args = merge_config(args, defaults, yaml_dict)
    return args
```

- [ ] **Step 6: Create the four ablation config YAML files**

`configs/baseline.yaml`:

```yaml
model: resnet50
epochs: 200
seed: 42
output-dir: results/baseline_seed42
```

`configs/cutmix.yaml`:

```yaml
model: resnet50
cutmix: true
epochs: 200
seed: 42
output-dir: results/cutmix_seed42
```

`configs/se.yaml`:

```yaml
model: se_resnet50
epochs: 200
seed: 42
output-dir: results/se_seed42
```

`configs/se_cutmix.yaml`:

```yaml
model: se_resnet50
cutmix: true
epochs: 200
seed: 42
output-dir: results/se_cutmix_seed42
```

- [ ] **Step 7: Add PyYAML to `requirements.txt`**

Append a line to `requirements.txt`:

```
PyYAML>=6.0
```

- [ ] **Step 8: Verify `train.py` still parses correctly**

```bash
pip install -r requirements.txt
python train.py --model resnet50 --config configs/baseline.yaml --epochs 1 --help
```

Expected: the `--help` output prints normally (confirms `build_parser()`
still returns a valid parser and `--config` is listed as an option). Also
run the full test suite:

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add config.py configs/ train.py requirements.txt tests/test_config.py
git commit -m "feat: add YAML config file support via --config"
```

---

### Task 6: Add MixUp augmentation

**Files:**
- Modify: `utils.py`
- Modify: `train.py`
- Modify: `tests/test_utils.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Consumes: `cutmix_data` from `utils.py` (existing).
- Produces: `mixup_data(x, y, alpha=1.0) -> Tuple[Tensor, Tensor, Tensor, float]` in `utils.py`; `apply_batch_augmentation(inputs, targets, args) -> Tuple[Tensor, Tensor, Tensor, float, bool]` in `train.py`. Task 8 (EMA) and Task 9 (`torch.compile`) will add more functions to `train.py` alongside this one but do not modify `apply_batch_augmentation`.

- [ ] **Step 1: Write the failing tests for `mixup_data`**

Append to `tests/test_utils.py`:

```python
from utils import mixup_data


def test_mixup_preserves_shape():
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    x_mixed, y_a, y_b, lam = mixup_data(x.clone(), y.clone(), alpha=1.0)
    assert x_mixed.shape == (4, 3, 32, 32)
    assert y_a.shape == (4,)
    assert y_b.shape == (4,)


def test_mixup_lambda_range():
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 100, (8,))
    for _ in range(50):
        _, _, _, lam = mixup_data(x.clone(), y.clone(), alpha=1.0)
        assert 0.0 <= lam <= 1.0


def test_mixup_blends_two_images_within_original_value_range():
    x = torch.zeros(2, 3, 4, 4)
    x[1] = 1.0
    y = torch.tensor([0, 1])
    x_mixed, _, _, lam = mixup_data(x.clone(), y.clone(), alpha=1.0)
    assert torch.all(x_mixed >= 0.0) and torch.all(x_mixed <= 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_utils.py -v
```

Expected: FAIL with `ImportError: cannot import name 'mixup_data' from 'utils'`.

- [ ] **Step 3: Implement `mixup_data` in `utils.py`**

Add below the existing `rand_bbox` function:

```python
def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    y_a, y_b = y, y[index]
    mixed_x = lam * x + (1 - lam) * x[index]

    return mixed_x, y_a, y_b, lam
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_utils.py -v
```

Expected: PASS for all `test_mixup_*` tests plus the pre-existing CutMix
tests.

- [ ] **Step 5: Write the failing tests for `apply_batch_augmentation`**

Append to `tests/test_train.py`:

```python
from argparse import Namespace

import torch

from train import apply_batch_augmentation


def make_aug_args(**overrides):
    base = {"cutmix": False, "mixup": False, "cutmix_prob": 0.5, "cutmix_alpha": 1.0, "mixup_alpha": 1.0}
    base.update(overrides)
    return Namespace(**base)


def test_no_augmentation_when_both_disabled():
    inputs = torch.randn(4, 3, 32, 32)
    targets = torch.tensor([0, 1, 2, 3])
    args = make_aug_args()
    out_inputs, targets_a, targets_b, lam, used_mixing = apply_batch_augmentation(inputs, targets, args)
    assert used_mixing is False
    assert lam == 1.0
    assert torch.equal(targets_a, targets)
    assert torch.equal(targets_b, targets)


def test_mixup_only_always_applies():
    inputs = torch.randn(4, 3, 32, 32)
    targets = torch.tensor([0, 1, 2, 3])
    args = make_aug_args(mixup=True)
    _, _, _, _, used_mixing = apply_batch_augmentation(inputs, targets, args)
    assert used_mixing is True


def test_cutmix_only_respects_zero_probability_gate():
    inputs = torch.randn(4, 3, 32, 32)
    targets = torch.tensor([0, 1, 2, 3])
    args = make_aug_args(cutmix=True, cutmix_prob=0.0)
    _, _, _, lam, used_mixing = apply_batch_augmentation(inputs, targets, args)
    assert used_mixing is False
    assert lam == 1.0


def test_both_enabled_uses_exactly_one_mixing_strategy():
    inputs = torch.randn(4, 3, 32, 32)
    targets = torch.tensor([0, 1, 2, 3])
    args = make_aug_args(cutmix=True, mixup=True, cutmix_prob=1.0)
    _, _, _, _, used_mixing = apply_batch_augmentation(inputs, targets, args)
    assert used_mixing is True
```

- [ ] **Step 6: Run tests to verify they fail**

```bash
pytest tests/test_train.py -v
```

Expected: FAIL with `ImportError: cannot import name 'apply_batch_augmentation' from 'train'`.

- [ ] **Step 7: Implement `apply_batch_augmentation` and wire it into `train_one_epoch`**

Add to `train.py` (near `train_one_epoch`), and add `--mixup`/`--mixup-alpha`
flags to `build_parser()`.

In `build_parser()`, add after the existing `--cutmix-alpha` argument:

```python
    parser.add_argument("--mixup", action="store_true", help="Enable MixUp augmentation")
    parser.add_argument("--mixup-alpha", type=float, default=1.0,
                        help="Beta distribution parameter for MixUp")
```

Update the import line at the top of `train.py`:

```python
from utils import cutmix_data, mixup_data, SoftLabelCrossEntropyLoss, set_seed
```

Add this new function above `train_one_epoch`:

```python
def apply_batch_augmentation(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, bool]:
    """Picks CutMix, MixUp, or neither for one batch. If both --cutmix and
    --mixup are enabled, a 50/50 coin flip decides which one runs this batch."""
    if args.cutmix and args.mixup:
        if np.random.rand() < 0.5:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, args.cutmix_alpha)
        else:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.mixup_alpha)
        return inputs, targets_a, targets_b, lam, True

    if args.cutmix and np.random.rand() < args.cutmix_prob:
        inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, args.cutmix_alpha)
        return inputs, targets_a, targets_b, lam, True

    if args.mixup:
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.mixup_alpha)
        return inputs, targets_a, targets_b, lam, True

    return inputs, targets, targets, 1.0, False
```

Then replace the augmentation branch inside `train_one_epoch`'s loop.
Replace:

```python
        with torch.amp.autocast(device.type, enabled=args.amp):
            r = np.random.rand(1)
            if args.cutmix and r < args.cutmix_prob:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, args.cutmix_alpha)
                outputs = model(inputs)
                loss = soft_criterion(outputs, targets_a, targets_b, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                targets_a = targets
                targets_b = targets
                lam = 1.0
```

with:

```python
        with torch.amp.autocast(device.type, enabled=args.amp):
            inputs, targets_a, targets_b, lam, used_mixing = apply_batch_augmentation(inputs, targets, args)
            outputs = model(inputs)
            loss = soft_criterion(outputs, targets_a, targets_b, lam) if used_mixing else criterion(outputs, targets)
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
pytest -v
```

Expected: all tests pass, including the 4 new `apply_batch_augmentation`
tests.

- [ ] **Step 9: Commit**

```bash
git add utils.py train.py tests/test_utils.py tests/test_train.py
git commit -m "feat: add MixUp augmentation with CutMix/MixUp batch selection"
```

---

### Task 7: Add label smoothing

**Files:**
- Modify: `train.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `build_criterion(args: argparse.Namespace) -> nn.CrossEntropyLoss` in `train.py`, used only inside `train_one_epoch`. No other task depends on this function.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_train.py`:

```python
from train import build_criterion


def test_label_smoothing_increases_loss_for_confident_correct_prediction():
    logits = torch.tensor([[5.0, 0.0, 0.0]])
    targets = torch.tensor([0])

    args_off = Namespace(label_smoothing=0.0)
    args_on = Namespace(label_smoothing=0.2)

    loss_off = build_criterion(args_off)(logits, targets)
    loss_on = build_criterion(args_on)(logits, targets)

    assert loss_on.item() > loss_off.item()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_train.py -v
```

Expected: FAIL with `ImportError: cannot import name 'build_criterion' from 'train'`.

- [ ] **Step 3: Add `--label-smoothing` flag and `build_criterion`**

In `build_parser()`, add after `--mixup-alpha`:

```python
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing factor for the training loss")
```

Add this function above `train_one_epoch` (near `apply_batch_augmentation`):

```python
def build_criterion(args: argparse.Namespace) -> nn.CrossEntropyLoss:
    return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
```

Inside `train_one_epoch`, replace:

```python
    criterion = nn.CrossEntropyLoss()
```

with:

```python
    criterion = build_criterion(args)
```

Leave `evaluate()`'s own `criterion = nn.CrossEntropyLoss()` (no smoothing)
unchanged — validation/test loss should stay unsmoothed for a fair,
comparable metric across configs.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_train.py -v
```

Expected: PASS.

- [ ] **Step 5: Run the full suite**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add train.py tests/test_train.py
git commit -m "feat: add label smoothing option for training loss"
```

---

### Task 8: Add EMA (Exponential Moving Average) of model weights

**Files:**
- Modify: `utils.py`
- Modify: `train.py`
- Modify: `tests/test_utils.py`

**Interfaces:**
- Consumes: `build_model` from `model.py` (test-only usage).
- Produces: `ModelEma` class in `utils.py` with `update(model)`, `state_dict()`, `load_state_dict(state_dict)`, `copy_to(model)`. `train_one_epoch`'s signature changes to accept an `ema: Optional[ModelEma] = None` keyword argument — this is the final signature later tasks (9) must call it with.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_utils.py`:

```python
from model import build_model
from utils import ModelEma


def test_model_ema_shadow_moves_toward_source_but_lags_behind():
    model = build_model("resnet50", num_classes=100)
    ema = ModelEma(model, decay=0.9)

    initial_shadow = {k: v.clone() for k, v in ema.state_dict().items()}

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    for _ in range(5):
        ema.update(model)

    shadow = ema.state_dict()
    model_state = model.state_dict()

    changed = any(
        not torch.equal(shadow[key], initial_shadow[key]) for key in shadow
    )
    still_lagging = any(
        shadow[key].dtype.is_floating_point and not torch.allclose(shadow[key], model_state[key])
        for key in shadow
    )

    assert changed, "EMA shadow should move after updates"
    assert still_lagging, "EMA shadow should not fully catch up to source after only 5 updates at decay=0.9"


def test_model_ema_state_dict_round_trips():
    model = build_model("resnet50", num_classes=100)
    ema = ModelEma(model, decay=0.99)
    ema.update(model)

    saved_state = {k: v.clone() for k, v in ema.state_dict().items()}

    new_ema = ModelEma(model, decay=0.99)
    new_ema.load_state_dict(saved_state)

    for key in saved_state:
        assert torch.equal(new_ema.state_dict()[key], saved_state[key])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_utils.py -v
```

Expected: FAIL with `ImportError: cannot import name 'ModelEma' from 'utils'`.

- [ ] **Step 3: Implement `ModelEma` in `utils.py`**

Add `import copy` to the top imports, then add the class at the end of the
file:

```python
class ModelEma:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for key, shadow_value in self.shadow.items():
            model_value = model_state[key]
            if shadow_value.dtype.is_floating_point:
                self.shadow[key] = self.decay * shadow_value + (1.0 - self.decay) * model_value
            else:
                self.shadow[key] = model_value.clone()

    def state_dict(self) -> dict:
        return self.shadow

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow = {k: v.clone() for k, v in state_dict.items()}

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_utils.py -v
```

Expected: PASS for both new `ModelEma` tests.

- [ ] **Step 5: Wire EMA into `train.py`**

In `build_parser()`, add after `--label-smoothing`:

```python
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="EMA decay for model weights (0 = disabled)")
```

Update the import line:

```python
from utils import cutmix_data, mixup_data, SoftLabelCrossEntropyLoss, set_seed, ModelEma
```

Change `train_one_epoch`'s signature from:

```python
def train_one_epoch(model, train_loader, optimizer, device, args, scaler=None):
```

to:

```python
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    scaler: Optional[torch.amp.GradScaler] = None,
    ema: Optional[ModelEma] = None,
) -> Tuple[float, float]:
```

Inside `train_one_epoch`, after both branches of the optimizer-step
`if scaler is not None: ... else: ...` block, add:

```python
        if ema is not None:
            ema.update(model)
```

In `main()`, after the model is built (`model = build_model(...).to(device)`),
add:

```python
    ema = ModelEma(model, decay=args.ema_decay) if args.ema_decay > 0 else None
```

Update the `train_one_epoch` call inside the training loop from:

```python
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, args, scaler)
```

to:

```python
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, args, scaler, ema)
```

After the existing `evaluate(model, val_loader, device, args.amp)` call in
the training loop, add EMA validation reporting. Replace:

```python
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, device, args.amp)
        scheduler.step()
```

with:

```python
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, device, args.amp)

        val_top1_ema = None
        if ema is not None:
            ema_model = build_model(args.model, num_classes=100).to(device)
            ema.copy_to(ema_model)
            _, val_top1_ema, _ = evaluate(ema_model, val_loader, device, args.amp)

        scheduler.step()
```

Update the CSV header write. Replace:

```python
    if not csv_exists:
        csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_top1", "val_top5", "lr", "time_s"])
```

with:

```python
    if not csv_exists:
        csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_top1", "val_top5",
                             "val_top1_ema", "lr", "time_s"])
```

Update the CSV row write. Replace:

```python
        csv_writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}",
                             f"{val_loss:.4f}", f"{val_top1:.2f}", f"{val_top5:.2f}",
                             f"{lr:.6f}", f"{elapsed:.1f}"])
```

with:

```python
        val_top1_ema_str = f"{val_top1_ema:.2f}" if val_top1_ema is not None else ""
        csv_writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}",
                             f"{val_loss:.4f}", f"{val_top1:.2f}", f"{val_top5:.2f}",
                             val_top1_ema_str, f"{lr:.6f}", f"{elapsed:.1f}"])
```

Add EMA state to checkpoints. Replace:

```python
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "config": config,
            "num_params": num_params,
        }
        if scaler is not None:
            checkpoint_state["scaler_state_dict"] = scaler.state_dict()
```

with:

```python
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "config": config,
            "num_params": num_params,
        }
        if scaler is not None:
            checkpoint_state["scaler_state_dict"] = scaler.state_dict()
        if ema is not None:
            checkpoint_state["ema_state_dict"] = ema.state_dict()
```

Restore EMA state on resume. Replace:

```python
            if scaler is not None and "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print(f"Resumed at epoch {start_epoch}, best_val_acc={best_val_acc:.2f}%")
```

with:

```python
            if scaler is not None and "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if ema is not None and "ema_state_dict" in checkpoint:
                ema.load_state_dict(checkpoint["ema_state_dict"])
            logger.info(f"Resumed at epoch {start_epoch}, best_val_acc={best_val_acc:.2f}%")
```

(Note: this resume block runs after `logger` is created since `setup_logging`
is called before the resume block in `main()`'s existing order — if that
ordering differs after Task 4, keep the `print` there as `logger.info`
consistent with the rest of Task 4's replacements.)

- [ ] **Step 6: Run the full suite**

```bash
pytest -v
```

Expected: all tests pass (EMA is off by default, so existing training-loop
tests are unaffected).

- [ ] **Step 7: Commit**

```bash
git add utils.py train.py tests/test_utils.py
git commit -m "feat: add EMA of model weights with checkpoint/resume support"
```

---

### Task 9: Add `torch.compile` opt-in flag

**Files:**
- Modify: `train.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `maybe_compile(model: nn.Module, args: argparse.Namespace) -> nn.Module` in `train.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_train.py`:

```python
from model import build_model
from train import maybe_compile


def test_maybe_compile_returns_model_unchanged_when_disabled():
    model = build_model("resnet50", num_classes=100)
    args = Namespace(compile=False)
    result = maybe_compile(model, args)
    assert result is model


def test_maybe_compile_wraps_model_when_enabled(monkeypatch):
    model = build_model("resnet50", num_classes=100)
    sentinel = object()
    monkeypatch.setattr(torch, "compile", lambda m: sentinel, raising=False)
    args = Namespace(compile=True)
    result = maybe_compile(model, args)
    assert result is sentinel
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_train.py -v
```

Expected: FAIL with `ImportError: cannot import name 'maybe_compile' from 'train'`.

- [ ] **Step 3: Add `--compile` flag and `maybe_compile`**

In `build_parser()`, add after `--ema-decay`:

```python
    parser.add_argument("--compile", action="store_true",
                        help="Wrap the model with torch.compile() if available")
```

Add this function near `apply_batch_augmentation`/`build_criterion`:

```python
def maybe_compile(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    if args.compile and hasattr(torch, "compile"):
        return torch.compile(model)
    return model
```

In `main()`, after the EMA setup line
(`ema = ModelEma(model, decay=args.ema_decay) if args.ema_decay > 0 else None`),
add:

```python
    model = maybe_compile(model, args)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_train.py -v
```

Expected: PASS for both new tests. (The monkeypatched `torch.compile` avoids
triggering a real, slow compilation during the test run.)

- [ ] **Step 5: Run the full suite**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add train.py tests/test_train.py
git commit -m "feat: add opt-in torch.compile flag"
```

---

### Task 10: Add batch inference support

**Files:**
- Modify: `inference.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- Consumes: `preprocess_image`, `CIFAR100_CLASSES` from `inference.py` (existing).
- Produces: `collect_image_paths(path: str) -> List[str]`, `preprocess_images(image_paths: List[str]) -> torch.Tensor`, `predict_batch(model, image_batch, device, top_k=5) -> List[List[Tuple[str, float]]]` in `inference.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_inference.py`:

```python
import torch
from PIL import Image

from inference import collect_image_paths, preprocess_images, predict_batch
from model import build_model


def test_collect_image_paths_single_file(tmp_path):
    img_path = tmp_path / "a.png"
    Image.new("RGB", (32, 32)).save(img_path)
    result = collect_image_paths(str(img_path))
    assert result == [str(img_path)]


def test_collect_image_paths_directory_filters_non_images(tmp_path):
    Image.new("RGB", (32, 32)).save(tmp_path / "a.png")
    Image.new("RGB", (32, 32)).save(tmp_path / "b.jpg")
    (tmp_path / "c.txt").write_text("not an image")

    result = collect_image_paths(str(tmp_path))
    assert len(result) == 2
    assert all(p.endswith((".png", ".jpg")) for p in result)


def test_preprocess_images_stacks_into_one_batch(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (32, 32)).save(p)
        paths.append(str(p))

    batch = preprocess_images(paths)
    assert batch.shape == (3, 3, 32, 32)


def test_predict_batch_returns_one_result_list_per_image():
    model = build_model("resnet50", num_classes=100)
    model.eval()
    device = torch.device("cpu")
    batch = torch.randn(2, 3, 32, 32)

    results = predict_batch(model, batch, device, top_k=3)

    assert len(results) == 2
    for image_results in results:
        assert len(image_results) == 3
        for class_name, prob in image_results:
            assert isinstance(class_name, str)
            assert 0.0 <= prob <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_inference.py -v
```

Expected: FAIL with `ImportError: cannot import name 'collect_image_paths' from 'inference'`.

- [ ] **Step 3: Implement the new functions in `inference.py`**

Update the imports at the top to include `List`, `Tuple` from `typing` and
`os` (already imported). Add these functions above `predict`:

```python
def collect_image_paths(path: str) -> List[str]:
    if os.path.isdir(path):
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        return sorted(
            os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(exts)
        )
    return [path]


def preprocess_images(image_paths: List[str]) -> torch.Tensor:
    tensors = [preprocess_image(p).squeeze(0) for p in image_paths]
    return torch.stack(tensors, dim=0)


def predict_batch(
    model: nn.Module, image_batch: torch.Tensor, device: torch.device, top_k: int = 5
) -> List[List[Tuple[str, float]]]:
    image_batch = image_batch.to(device)
    with torch.no_grad():
        outputs = model(image_batch)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = probabilities.topk(top_k, dim=1)

    batch_results = []
    for b in range(image_batch.size(0)):
        results = []
        for i in range(top_k):
            idx = top_indices[b][i].item()
            prob = top_probs[b][i].item()
            results.append((CIFAR100_CLASSES[idx], prob))
        batch_results.append(results)
    return batch_results
```

Update `main()`'s argument help text and body. Replace:

```python
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
```

with:

```python
    parser.add_argument('--image', type=str, required=True,
                        help='Path to a single input image, or a directory of images')
```

Then replace:

```python
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        return

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device, args.model)
    image_tensor = preprocess_image(args.image)
    results = predict(model, image_tensor, device, top_k=args.top_k)

    print(f"\nPrediction results for: {args.image}")
    print("-" * 40)
    for rank, (class_name, prob) in enumerate(results, 1):
        print(f"  #{rank}: {class_name:20s} ({prob*100:.2f}%)")
```

with:

```python
    if not os.path.exists(args.image):
        print(f"Error: Image path '{args.image}' not found.")
        return

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device, args.model)
    image_paths = collect_image_paths(args.image)
    if not image_paths:
        print(f"Error: No images found at '{args.image}'.")
        return

    image_batch = preprocess_images(image_paths)
    batch_results = predict_batch(model, image_batch, device, top_k=args.top_k)

    for image_path, results in zip(image_paths, batch_results):
        print(f"\nPrediction results for: {image_path}")
        print("-" * 40)
        for rank, (class_name, prob) in enumerate(results, 1):
            print(f"  #{rank}: {class_name:20s} ({prob*100:.2f}%)")
```

The existing single-image `predict()` function stays as-is (still usable
programmatically); `main()` now uses `predict_batch()` for both single files
and directories.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_inference.py -v
```

Expected: PASS for all 4 tests.

- [ ] **Step 5: Run the full suite**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add inference.py tests/test_inference.py
git commit -m "feat: support batch inference over a directory of images"
```

---

### Task 11: Add ONNX export with parity check

**Files:**
- Create: `export_onnx.py`
- Modify: `requirements-dev.txt`
- Create: `tests/test_export_onnx.py`

**Interfaces:**
- Consumes: `load_model` from `inference.py` (existing).
- Produces: `export_to_onnx(model, output_path) -> None`, `verify_onnx_parity(model, onnx_path, atol=1e-4) -> bool` in `export_onnx.py`.

- [ ] **Step 1: Add ONNX deps to `requirements-dev.txt`**

Append two lines:

```
onnx>=1.15.0
onnxruntime>=1.17.0
```

Install them:

```bash
pip install -r requirements-dev.txt
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_export_onnx.py`:

```python
import os

from export_onnx import export_to_onnx, verify_onnx_parity
from model import build_model


def test_export_and_parity_check(tmp_path):
    model = build_model("resnet50", num_classes=100)
    model.eval()
    output_path = os.path.join(str(tmp_path), "model.onnx")

    export_to_onnx(model, output_path)

    assert os.path.isfile(output_path)
    assert verify_onnx_parity(model, output_path, atol=1e-4)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_export_onnx.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'export_onnx'`.

- [ ] **Step 4: Create `export_onnx.py`**

```python
import argparse

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from inference import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="model.onnx")
    parser.add_argument("--model", type=str, default="se_resnet50",
                        choices=["resnet50", "se_resnet50"])
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def export_to_onnx(model: nn.Module, output_path: str) -> None:
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
    )


def verify_onnx_parity(model: nn.Module, onnx_path: str, atol: float = 1e-4) -> bool:
    model.eval()
    test_input = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        torch_output = model(test_input).numpy()

    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(None, {"input": test_input.numpy()})[0]

    return bool(np.allclose(torch_output, onnx_output, atol=atol))


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    model = load_model(args.checkpoint, device, args.model)

    export_to_onnx(model, args.output)
    print(f"Exported ONNX model to: {args.output}")

    if verify_onnx_parity(model, args.output, atol=args.atol):
        print(f"Parity check passed (atol={args.atol})")
    else:
        raise RuntimeError("ONNX output diverges from PyTorch output beyond tolerance")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_export_onnx.py -v
```

Expected: PASS.

- [ ] **Step 6: Run the full suite**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add export_onnx.py requirements-dev.txt tests/test_export_onnx.py
git commit -m "feat: add ONNX export script with numerical parity check"
```

---

### Task 12: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `README_KR.md`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: nothing consumed by other tasks — this is the final task.

- [ ] **Step 1: Update `README.md`'s CLI options table**

Replace the existing table:

```markdown
| Flag | Default | Description |
|------|---------|-------------|
| `--model` | se_resnet50 | Architecture: `resnet50` or `se_resnet50` |
| `--epochs` | 200 | Total training epochs |
| `--batch-size` | 128 | Mini-batch size |
| `--lr` | 0.1 | Initial learning rate |
| `--cutmix` | off | Enable CutMix augmentation |
| `--cutmix-prob` | 0.5 | Probability of applying CutMix per batch |
| `--cutmix-alpha` | 1.0 | Beta distribution parameter |
| `--seed` | 42 | Random seed for reproducibility |
| `--output-dir` | runs/default | Output directory for checkpoints and logs |
| `--resume` | - | Path to checkpoint to resume from |
| `--amp` | off | Enable automatic mixed precision |
| `--val-split` | 0.1 | Fraction of training data for validation |
| `--num-workers` | 4 | DataLoader workers |
| `--grad-clip` | 1.0 | Max gradient norm |
```

with:

```markdown
| Flag | Default | Description |
|------|---------|-------------|
| `--config` | - | Path to a YAML config file (CLI flags override YAML values) |
| `--model` | se_resnet50 | Architecture: `resnet50` or `se_resnet50` |
| `--epochs` | 200 | Total training epochs |
| `--batch-size` | 128 | Mini-batch size |
| `--lr` | 0.1 | Initial learning rate |
| `--cutmix` | off | Enable CutMix augmentation |
| `--cutmix-prob` | 0.5 | Probability of applying CutMix per batch |
| `--cutmix-alpha` | 1.0 | Beta distribution parameter |
| `--mixup` | off | Enable MixUp augmentation |
| `--mixup-alpha` | 1.0 | Beta distribution parameter for MixUp |
| `--label-smoothing` | 0.0 | Label smoothing factor for the training loss |
| `--ema-decay` | 0.0 | EMA decay for model weights (0 = disabled) |
| `--compile` | off | Wrap the model with `torch.compile()` if available |
| `--seed` | 42 | Random seed for reproducibility |
| `--output-dir` | runs/default | Output directory for checkpoints and logs |
| `--resume` | - | Path to checkpoint to resume from |
| `--amp` | off | Enable automatic mixed precision |
| `--val-split` | 0.1 | Fraction of training data for validation |
| `--num-workers` | 4 | DataLoader workers |
| `--grad-clip` | 1.0 | Max gradient norm |

If both `--cutmix` and `--mixup` are enabled, each batch uses one or the
other (chosen by a 50/50 coin flip), never both at once.
```

- [ ] **Step 2: Add a "Configuration Files" section to `README.md`**

Insert after the "### Full Ablation Study" section and before "### Generate
Comparison Plots":

```markdown
### Using a Config File

Instead of passing every flag on the command line, use one of the presets
in `configs/` (these match the 4 ablation study configs exactly):

```bash
python train.py --config configs/se_cutmix.yaml
```

Any CLI flag passed alongside `--config` overrides the corresponding YAML
value:

```bash
python train.py --config configs/se_cutmix.yaml --epochs 5
```
```

- [ ] **Step 3: Add "ONNX Export" and update "Inference" sections in `README.md`**

Replace:

```markdown
### Inference

```bash
python inference.py --image <image_path> --checkpoint runs/se_cutmix_seed42/best.pth
```
```

with:

```markdown
### Inference

Single image:

```bash
python inference.py --image <image_path> --checkpoint runs/se_cutmix_seed42/best.pth
```

Or a whole directory of images at once (batched into one forward pass):

```bash
python inference.py --image <directory_path> --checkpoint runs/se_cutmix_seed42/best.pth
```

### ONNX Export

Export a checkpoint to ONNX and verify numerical parity against the
original PyTorch model:

```bash
python export_onnx.py --checkpoint runs/se_cutmix_seed42/best.pth --output model.onnx
```
```

- [ ] **Step 4: Update the "Tests" and "Project Structure" sections in `README.md`**

Replace:

```markdown
## Tests

```bash
python3 tests/test_model.py
```
```

with:

```markdown
## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

CI runs the same suite plus `ruff check .` on every push/PR (see
`.github/workflows/ci.yml`).
```

Replace the project structure tree:

```markdown
```
enhanced-resnet50-cv/
├── model.py             # ResNet50 and SE-ResNet50 architecture
├── train.py             # Training pipeline with CLI, validation split, logging
├── utils.py              # CutMix, soft label loss, seed utilities
├── inference.py          # Single image inference
├── plot_results.py       # Aggregate results and generate comparison plots
├── run_experiments.sh    # Full ablation experiment script
├── tests/
│   └── test_model.py     # Unit tests for model, CutMix, checkpoints
├── requirements.txt      # Dependencies
├── assets/               # Visualization assets
├── README.md             # English documentation
└── README_KR.md          # Korean documentation
```
```

with:

```markdown
```
enhanced-resnet50-cv/
├── model.py              # ResNet50 and SE-ResNet50 architecture
├── train.py              # Training pipeline with CLI, validation split, logging
├── utils.py              # CutMix, MixUp, soft label loss, EMA, seed utilities
├── inference.py          # Single-image and batch inference
├── export_onnx.py        # ONNX export with numerical parity check
├── config.py             # YAML config loading/merging for train.py
├── configs/              # YAML presets matching the 4 ablation configs
├── plot_results.py       # Aggregate results and generate comparison plots
├── run_experiments.sh    # Full ablation experiment script
├── tests/
│   ├── conftest.py        # Shared pytest path setup
│   ├── test_model.py       # Model architecture tests
│   ├── test_utils.py       # CutMix/MixUp/EMA/checkpoint tests
│   ├── test_train.py       # Training-loop helper tests
│   ├── test_inference.py   # Batch inference tests
│   ├── test_config.py      # Config load/merge tests
│   └── test_export_onnx.py # ONNX export parity test
├── .github/workflows/ci.yml  # CI: lint + test on every push/PR
├── requirements.txt      # Runtime dependencies
├── requirements-dev.txt  # Dev/test/lint dependencies
├── pytest.ini            # pytest configuration
├── ruff.toml             # Lint configuration
├── mypy.ini              # Optional local type-checking configuration
├── assets/               # Visualization assets
├── README.md             # English documentation
└── README_KR.md          # Korean documentation
```
```

- [ ] **Step 5: Add a "Future Work" section to `README.md`**

Insert before the "## References" section:

```markdown
## Future Work

- **Multi-GPU / DDP training** — the current pipeline targets a single
  GPU/CPU; distributed data-parallel training would be the natural next
  step for running the full ablation study at scale.
- **Additional attention modules** — CBAM and ECA are natural companions
  to the existing SE-Block for a broader architecture ablation.
```

- [ ] **Step 6: Mirror all of the above changes in `README_KR.md`**

Apply the equivalent Korean-language edits to `README_KR.md`: translate the
new CLI flag rows, add "설정 파일 사용" (Configuration Files), "ONNX 내보내기"
(ONNX Export), update "추론" (Inference) with the directory-batch example,
update "테스트" (Tests) to use `pytest`, update "프로젝트 구조" (Project
Structure) tree to match the new file list, and add a "향후 작업" (Future
Work) section listing the same two deferred items (multi-GPU/DDP, CBAM/ECA).
Keep the existing Korean tone/style used elsewhere in the file.

- [ ] **Step 7: Verify the full suite still passes (docs-only task, but confirm nothing regressed)**

```bash
pytest -v
ruff check .
```

Expected: all tests pass, no lint errors.

- [ ] **Step 8: Commit**

```bash
git add README.md README_KR.md
git commit -m "docs: document config files, new training flags, ONNX export, and batch inference"
```

---

## Self-Review Notes

- **Spec coverage:** pytest migration (Task 1), CI (Task 2), type hints
  (Task 3), logging (Task 4), config management (Task 5), MixUp (Task 6),
  label smoothing (Task 7), EMA (Task 8), `torch.compile` (Task 9), batch
  inference (Task 10), ONNX export (Task 11), documentation (Task 12) — all
  three spec sections and their sub-bullets are covered.
- **Placeholder scan:** no TBD/TODO; every step has complete, runnable code.
- **Type consistency:** `apply_batch_augmentation` (Task 6) returns
  `Tuple[Tensor, Tensor, Tensor, float, bool]` consistently in its
  definition, its test assertions, and its call site in `train_one_epoch`.
  `ModelEma` (Task 8)'s `state_dict()`/`load_state_dict()`/`copy_to()` names
  match between the class definition, its tests, and its usage in `main()`.
  `train_one_epoch`'s signature gains `ema` in Task 8 and is called with the
  updated signature in the same task — Task 9 does not touch this signature.
