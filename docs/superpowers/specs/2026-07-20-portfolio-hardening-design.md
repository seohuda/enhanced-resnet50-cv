# Enhanced ResNet50 CV — Portfolio Hardening Design

## Context

`enhanced-resnet50-cv` is a PyTorch project implementing ResNet50 / SE-ResNet50
on CIFAR-100 with an optional CutMix augmentation, structured as a 4-config
ablation study (baseline / CutMix / SE / SE+CutMix, 3 seeds each). The
codebase already has a clean architecture (`model.py`, `train.py`, `utils.py`,
`inference.py`), reproducibility guarantees (fixed seeds, deterministic
cudnn), checkpoint/resume support, and a small hand-rolled test suite
(`tests/test_model.py`).

**Purpose of this round of work**: the repo's primary use is as a
portfolio / hiring demo, and there are no near-term plans to actually run
training on real hardware. That reprioritizes the work away from
infrastructure that only pays off at scale (multi-GPU/DDP) and toward things
that read well in a code review: test coverage, CI, config management, a
couple of well-chosen ML technique additions, and lightweight
performance/export polish.

**Non-goals**: multi-GPU/DDP training, additional attention modules beyond
SE (CBAM/ECA), new ablation ordering — explicitly deferred as scope-creep
beyond "core, but solid." DDP is noted as a documented future-work line in
the README rather than built.

## Scope

Three areas, sized to land as one cohesive PR-able body of work:

1. Engineering foundation (tests, config, CI, logging, type hints)
2. Model/training technique additions (MixUp, label smoothing, EMA)
3. Performance/export polish (ONNX export, batch inference, `torch.compile`)

All additions are opt-in / additive — existing CLI flags, `run_experiments.sh`,
and the 4 existing ablation configs continue to behave exactly as before
unless a user opts into a new flag.

## 1. Engineering Foundation

### Testing
- Convert `tests/test_model.py` from the current hand-rolled runner to real
  `pytest` (plain `assert`, no custom pass/fail loop). Parametrize the
  model-shape tests (`resnet50` / `se_resnet50`) instead of duplicating them.
- Add `tests/test_utils.py` covering CutMix, MixUp, label smoothing loss, and
  EMA (new in section 2).
- Add `tests/test_config.py` covering the new YAML config loader (section
  below): CLI-overrides-YAML precedence, unknown-key handling, defaults.
- Add `pytest`, `pytest-cov` to `requirements.txt` (dev-only deps could live
  in a `requirements-dev.txt` instead — see open question below, resolved as:
  separate `requirements-dev.txt` to keep the runtime install lean).

### Config management
- New `config.py`: a typed dataclass (`TrainConfig`) mirroring `train.py`'s
  argparse fields, plus a `load_config(path) -> dict` that reads YAML and a
  `merge_config(args, yaml_dict)` that applies YAML values only where the CLI
  left the argparse default untouched (explicit CLI flags win).
- New `configs/` directory with one YAML per existing ablation entry:
  `baseline.yaml`, `cutmix.yaml`, `se.yaml`, `se_cutmix.yaml` — values taken
  from current `run_experiments.sh` invocations, so behavior is unchanged.
- `train.py` gains `--config <path>` (optional). No `--config` = today's
  exact behavior.

### CI
- `.github/workflows/ci.yml`: on push/PR, Python 3.10, ubuntu-latest, CPU
  only. Steps: install `requirements.txt` + `requirements-dev.txt`, `ruff
  check .`, `pytest --cov`.
- No CUDA/GPU runner — all tests operate on tiny random tensors, not real
  training, so CPU is sufficient and keeps CI fast/free.

### Logging
- Replace `print()` calls in `train.py` with the standard `logging` module:
  a console handler (same human-readable table format as today) plus a file
  handler writing to `<output-dir>/train.log`. `inference.py` and
  `plot_results.py` keep `print()` since they're one-shot CLI scripts, not
  long-running training loops — logging there would be pure noise.

### Type hints
- Add parameter/return type hints across `model.py`, `utils.py`, `train.py`,
  `inference.py`, `config.py`. No behavioral change.
- Add a permissive `mypy.ini` (ignore missing imports for `torch`/`torchvision`)
  so `mypy .` is runnable locally, but it is **not** wired into CI as a
  blocking gate — torch's typing stubs are inconsistent enough that a hard
  gate would produce noisy false positives rather than signal.

## 2. Model/Training Technique Additions

### MixUp
- `mixup_data(x, y, alpha=1.0)` in `utils.py`, same return contract as the
  existing `cutmix_data`: `(x_mixed, y_a, y_b, lam)`, reusing
  `SoftLabelCrossEntropyLoss`.
- `train.py` gains `--mixup` (flag) and `--mixup-alpha` (default 1.0,
  matching CutMix's convention).
- If both `--cutmix` and `--mixup` are enabled, `train_one_epoch` picks one
  per batch via a coin flip (50/50) — documented in the CLI help string and
  the README, not hidden behavior.

### Label Smoothing
- `--label-smoothing` (float, default `0.0`) passed straight into
  `nn.CrossEntropyLoss(label_smoothing=...)` on the non-mixed path. Simple,
  well-understood, cheap to add, easy to sanity-test (loss for
  a one-hot-confident prediction should be strictly higher than with
  smoothing off).

### EMA (Exponential Moving Average) of weights
- `ModelEma` class in `utils.py`: holds a shadow copy of `state_dict()`,
  updated after every optimizer step via
  `shadow = decay * shadow + (1 - decay) * param`.
- `--ema-decay` (default `0.0` = disabled).
- Checkpoints gain an `ema_state_dict` key when enabled; `evaluate()` is
  called against both the raw and EMA weights when EMA is on, and both
  numbers are logged/written to `metrics.csv` (`val_top1`, `val_top1_ema`).
- Unit test: after N synthetic updates with a fixed decay, shadow params
  should have moved measurably toward source params without becoming
  identical (sanity check on the EMA math itself, not a convergence claim).

None of the three techniques change existing default behavior — all default
off, so the 4 existing ablation configs and `run_experiments.sh` remain
byte-for-byte equivalent runs.

## 3. Performance/Export Polish

### ONNX export
- New `export_onnx.py`: loads a checkpoint (reusing `inference.load_model`'s
  logic for model-name auto-detection), exports via `torch.onnx.export` with
  a dynamic batch axis, then runs a parity check — same input through the
  PyTorch model and through `onnxruntime.InferenceSession`, asserting
  `torch.allclose` within a documented tolerance (`atol=1e-4`). Fails loudly
  if outputs diverge, so it's a real correctness check rather than "it
  exported without an exception."
- Adds `onnx`, `onnxruntime` to `requirements-dev.txt` (not needed for
  training/inference, only for this optional export path).

### Batch inference
- `inference.py`'s `--image` flag accepts either a single file or a
  directory; when given a directory, all images are preprocessed and
  stacked into one batch tensor for a single forward pass (rather than
  looping one-image-at-a-time), with per-image top-k results printed same
  as today.

### `torch.compile` opt-in
- `--compile` flag in `train.py`. When set, wraps the model with
  `torch.compile(model)` guarded by `hasattr(torch, "compile")` so it
  degrades gracefully on older PyTorch installs instead of erroring.

DDP/multi-GPU training is explicitly out of scope for this round — the
README will gain a short "Future Work" bullet noting it as a natural next
step if training moves to a multi-GPU environment.

## Testing Strategy Summary

- `pytest` end to end (replacing the current hand-rolled runner).
- New coverage: MixUp, label smoothing loss behavior, EMA math, config
  load/merge precedence, ONNX export parity (marked `slow`/optional if it
  meaningfully lengthens CI — decided at implementation time based on actual
  runtime).
- CI runs the full suite (minus anything marked slow) on every push/PR,
  CPU-only.

## Documentation

- `README.md` / `README_KR.md` updated: new CLI flags table rows
  (`--mixup`, `--mixup-alpha`, `--label-smoothing`, `--ema-decay`,
  `--compile`, `--config`), new `configs/` usage note, new `export_onnx.py`
  usage section, "Future Work" bullet for DDP/multi-GPU and additional
  attention modules (CBAM/ECA) as explicitly deferred ideas.
