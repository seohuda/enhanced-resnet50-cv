# Enhanced ResNet50 with SE-Block and CutMix for CIFAR-100

[![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4%EB%A1%9C%20%EC%9D%BD%EA%B8%B0-blue)](README_KR.md)

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![TorchVision](https://img.shields.io/badge/TorchVision-0.15%2B-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557C?logo=python&logoColor=white)

## Project Overview

This project implements and compares ResNet50 variants on CIFAR-100 image classification, applying two improvement strategies:

1. **Structural Improvement**: Integrating Squeeze-and-Excitation (SE) attention modules into each Bottleneck block.
2. **Data-Centric Improvement**: Applying CutMix data augmentation with Soft Label Cross-Entropy Loss.

A full ablation study (4 configurations x 3 seeds) is provided to separately measure the contribution of each technique.

## Key Techniques

### SE-Block (Squeeze-and-Excitation)

The SE-Block adaptively recalibrates channel-wise feature responses by modeling interdependencies between channels. It is inserted after the third convolution in each Bottleneck block, before the residual addition.

- **Squeeze**: Global Average Pooling compresses spatial dimensions into a channel descriptor.
- **Excitation**: Two FC layers learn channel-wise attention weights.
- **Scale**: The original feature map is rescaled by the learned weights.

### CutMix Augmentation

CutMix cuts a random patch from one training image and pastes it onto another, mixing labels proportionally to the patch area. This encourages the model to focus on less discriminative parts and reduces overfitting.

## Experiment Design

### Data Split

| Split | Size | Usage |
|-------|------|-------|
| Train | 45,000 | Model training |
| Validation | 5,000 | Epoch selection, hyperparameter tuning |
| Test | 10,000 | Final evaluation (once, after all training) |

The validation set is split from CIFAR-100 training data using a fixed seed. The test set is only evaluated after training completes to avoid data leakage.

### Ablation Configurations

| Experiment | Model | CutMix | Purpose |
|-----------|-------|--------|---------|
| Baseline | ResNet50 | Off | Reference performance |
| CutMix only | ResNet50 | On | Isolate CutMix contribution |
| SE only | SE-ResNet50 | Off | Isolate SE-Block contribution |
| SE + CutMix | SE-ResNet50 | On | Combined effect |

Each configuration is run with 3 seeds (42, 123, 456) and results are reported as mean +/- std.

### Reproducibility

- Random seeds are fixed (Python, NumPy, PyTorch, CUDA)
- `torch.backends.cudnn.deterministic = True`
- All hyperparameters are logged to `config.json` per run
- Per-epoch metrics are saved to `metrics.csv`
- Checkpoints include full training state (model, optimizer, scheduler, epoch)

## How to Run

### Requirements

```bash
pip install -r requirements.txt
```

### Single Training Run

```bash
python train.py \
  --model se_resnet50 \
  --cutmix \
  --epochs 200 \
  --seed 42 \
  --output-dir runs/se_cutmix_seed42
```

### Full Ablation Study

```bash
./run_experiments.sh
```

This runs all 4 configurations x 3 seeds (12 training runs total).

### Generate Comparison Plots

```bash
python plot_results.py --results-dir results
```

### Resume Training

```bash
python train.py --resume runs/se_cutmix_seed42/last.pth
```

### Inference

```bash
python inference.py --image <image_path> --checkpoint runs/se_cutmix_seed42/best.pth
```

### CLI Options

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

## Architecture

```
Input (3x32x32)
  +-- Conv1 (3x3, stride=1) + BN + ReLU
  +-- Layer1: 3x Bottleneck (64)  [+SE if se_resnet50]
  +-- Layer2: 4x Bottleneck (128) [+SE if se_resnet50]
  +-- Layer3: 6x Bottleneck (256) [+SE if se_resnet50]
  +-- Layer4: 3x Bottleneck (512) [+SE if se_resnet50]
  +-- AdaptiveAvgPool2d -> FC (100 classes)
```

## Output Structure

After running experiments:

```
results/
├── baseline_seed42/
│   ├── config.json
│   ├── metrics.csv
│   ├── summary.json
│   ├── best.pth
│   └── last.pth
├── cutmix_seed42/
├── se_seed42/
├── se_cutmix_seed42/
├── ...  (repeat for seeds 123, 456)
└── comparison.png
```

## Tests

```bash
python3 tests/test_model.py
```

## Project Structure

```
enhanced-resnet50-cv/
├── model.py             # ResNet50 and SE-ResNet50 architecture
├── train.py             # Training pipeline with CLI, validation split, logging
├── utils.py             # CutMix, soft label loss, seed utilities
├── inference.py         # Single image inference
├── plot_results.py      # Aggregate results and generate comparison plots
├── run_experiments.sh   # Full ablation experiment script
├── tests/
│   └── test_model.py    # Unit tests for model, CutMix, checkpoints
├── requirements.txt     # Dependencies
├── assets/              # Visualization assets
├── README.md            # English documentation
└── README_KR.md         # Korean documentation
```

## References

- [Squeeze-and-Excitation Networks (Hu et al., 2018)](https://arxiv.org/abs/1709.01507)
- [CutMix: Regularization Strategy to Train Strong Classifiers (Yun et al., 2019)](https://arxiv.org/abs/1905.04899)
- [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385)
