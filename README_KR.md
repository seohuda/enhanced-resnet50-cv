# Enhanced ResNet50 with SE-Block and CutMix for CIFAR-100

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![TorchVision](https://img.shields.io/badge/TorchVision-0.15%2B-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557C?logo=python&logoColor=white)

## 프로젝트 개요

CIFAR-100 이미지 분류에서 ResNet50 변형 모델들을 구현하고, 두 가지 개선 기법의 효과를 비교합니다.

1. **구조적 개선**: 각 Bottleneck 블록에 Squeeze-and-Excitation (SE) 어텐션 모듈 통합
2. **데이터 중심 개선**: CutMix 데이터 증강과 Soft Label Cross-Entropy Loss 적용

4가지 구성(Baseline / CutMix / SE / SE+CutMix) x 3 seeds로 ablation study를 수행하여 각 기법의 기여도를 분리 측정합니다.

## 핵심 기법

### SE-Block (Squeeze-and-Excitation)

채널 간 상호 의존성을 모델링하여 채널별 특성 응답을 적응적으로 재보정합니다. 각 Bottleneck의 3번째 conv 뒤, residual 합산 전에 삽입됩니다.

- **Squeeze**: GAP으로 공간 차원을 채널 디스크립터로 압축
- **Excitation**: 두 FC 층으로 채널별 어텐션 가중치 학습
- **Scale**: 학습된 가중치로 원본 특성 맵 재조정

### CutMix

한 학습 이미지에서 랜덤 패치를 잘라 다른 이미지에 붙이고, 패치 면적에 비례하여 레이블을 혼합합니다. 덜 판별적인 영역에도 주목하도록 유도하며 과적합을 줄입니다.

## 실험 설계

### 데이터 분할

| 분할 | 크기 | 용도 |
|------|------|------|
| Train | 45,000 | 모델 학습 |
| Validation | 5,000 | epoch 선택, 하이퍼파라미터 튜닝 |
| Test | 10,000 | 최종 평가 (학습 완료 후 1회만) |

Validation은 CIFAR-100 학습 데이터에서 고정된 seed로 분리합니다. Test set은 모든 학습이 끝난 뒤 한 번만 평가하여 데이터 누수를 방지합니다.

### Ablation 구성

| 실험 | 모델 | CutMix | 목적 |
|------|------|--------|------|
| Baseline | ResNet50 | X | 기준 성능 |
| CutMix only | ResNet50 | O | CutMix 기여도 분리 |
| SE only | SE-ResNet50 | X | SE-Block 기여도 분리 |
| SE + CutMix | SE-ResNet50 | O | 결합 효과 |

각 구성은 3개 seed(42, 123, 456)로 반복하며 mean +/- std로 보고합니다.

### 재현성

- Random seed 고정 (Python, NumPy, PyTorch, CUDA)
- `torch.backends.cudnn.deterministic = True`
- 모든 하이퍼파라미터를 `config.json`으로 기록
- epoch별 메트릭을 `metrics.csv`로 저장
- 체크포인트에 전체 학습 상태 포함 (model, optimizer, scheduler, epoch)

## 실행 방법

### 환경 설정

```bash
pip install -r requirements.txt
```

### 단일 학습

```bash
python train.py \
  --model se_resnet50 \
  --cutmix \
  --epochs 200 \
  --seed 42 \
  --output-dir runs/se_cutmix_seed42
```

### 전체 Ablation Study

```bash
./run_experiments.sh
```

4 구성 x 3 seeds = 12회 학습을 순차 실행합니다.

### 결과 시각화

```bash
python plot_results.py --results-dir results
```

### 학습 재개

```bash
python train.py --resume runs/se_cutmix_seed42/last.pth
```

### 추론

```bash
python inference.py --image <이미지 경로> --checkpoint runs/se_cutmix_seed42/best.pth
```

### CLI 옵션

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--model` | se_resnet50 | 아키텍처: `resnet50` 또는 `se_resnet50` |
| `--epochs` | 200 | 전체 학습 에포크 |
| `--batch-size` | 128 | 미니배치 크기 |
| `--lr` | 0.1 | 초기 학습률 |
| `--cutmix` | off | CutMix 활성화 |
| `--cutmix-prob` | 0.5 | 배치별 CutMix 적용 확률 |
| `--cutmix-alpha` | 1.0 | Beta 분포 파라미터 |
| `--seed` | 42 | 재현성을 위한 랜덤 시드 |
| `--output-dir` | runs/default | 체크포인트, 로그 저장 디렉토리 |
| `--resume` | - | 재개할 체크포인트 경로 |
| `--amp` | off | Automatic Mixed Precision 활성화 |
| `--val-split` | 0.1 | validation 비율 |
| `--num-workers` | 4 | DataLoader 워커 수 |
| `--grad-clip` | 1.0 | gradient norm 상한 |

## 아키텍처

```
Input (3x32x32)
  +-- Conv1 (3x3, stride=1) + BN + ReLU
  +-- Layer1: 3x Bottleneck (64)  [+SE if se_resnet50]
  +-- Layer2: 4x Bottleneck (128) [+SE if se_resnet50]
  +-- Layer3: 6x Bottleneck (256) [+SE if se_resnet50]
  +-- Layer4: 3x Bottleneck (512) [+SE if se_resnet50]
  +-- AdaptiveAvgPool2d -> FC (100 classes)
```

## 출력 구조

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
├── ...  (seed 123, 456 반복)
└── comparison.png
```

## 테스트

```bash
python3 tests/test_model.py
```

## 프로젝트 구조

```
enhanced-resnet50-cv/
├── model.py             # ResNet50, SE-ResNet50 아키텍처
├── train.py             # CLI, validation 분리, 로깅 포함 학습 파이프라인
├── utils.py             # CutMix, soft label loss, seed 유틸리티
├── inference.py         # 단일 이미지 추론
├── plot_results.py      # 결과 집계 및 비교 그래프 생성
├── run_experiments.sh   # 전체 ablation 실험 스크립트
├── tests/
│   └── test_model.py    # 모델, CutMix, 체크포인트 단위 테스트
├── requirements.txt     # 의존성
├── assets/              # 시각화 자료
├── README.md            # 영문 문서
└── README_KR.md         # 한국어 문서
```

## 참고 문헌

- [Squeeze-and-Excitation Networks (Hu et al., 2018)](https://arxiv.org/abs/1709.01507)
- [CutMix: Regularization Strategy to Train Strong Classifiers (Yun et al., 2019)](https://arxiv.org/abs/1905.04899)
- [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385)
