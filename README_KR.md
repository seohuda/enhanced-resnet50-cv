# Enhanced ResNet50 with SE-Block and CutMix for CIFAR-100

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![TorchVision](https://img.shields.io/badge/TorchVision-0.15%2B-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557C?logo=python&logoColor=white)

## 프로젝트 개요

본 프로젝트는 CIFAR-100 이미지 분류를 위해 ResNet50 아키텍처를 두 가지 핵심 전략으로 개선합니다:

1. **구조적 개선**: ResNet50의 각 Bottleneck 블록에 Squeeze-and-Excitation (SE) 어텐션 모듈 통합
2. **데이터 중심 개선**: CutMix 데이터 증강과 Soft Label Cross-Entropy Loss 적용

채널별 어텐션 재보정(SE-Block)과 영역 수준 믹싱 정규화(CutMix)의 조합으로 기본 ResNet50 대비 향상된 일반화 성능과 분류 정확도를 달성합니다.

## 핵심 개선 사항

### SE-Block (Squeeze-and-Excitation)

SE-Block은 채널 간 상호 의존성을 명시적으로 모델링하여 채널별 특성 응답을 적응적으로 재보정합니다. 각 Bottleneck 블록의 3번째 컨볼루션 이후, 잔차 연결 이전에 삽입됩니다.

**메커니즘:**
- **Squeeze**: Global Average Pooling으로 공간 차원을 채널 디스크립터로 압축
- **Excitation**: 두 개의 완전연결 층으로 채널별 어텐션 가중치 학습
- **Scale**: 학습된 어텐션 가중치로 원본 특성 맵 재조정

### CutMix 증강

CutMix는 하나의 학습 이미지에서 랜덤 패치를 잘라 다른 이미지에 붙이고, 패치 면적에 비례하여 레이블을 혼합합니다. 이를 통해 모델이:

- 객체의 덜 판별적인 부분에도 집중
- 가려짐(occlusion)에 대한 강건성 향상
- 더 강한 정규화로 과적합 감소

학습 루프에서 혼합된 레이블을 적절히 처리하기 위해 Soft Label Cross-Entropy Loss를 사용합니다.

## 성능 비교

| 모델 | Top-1 정확도 | Top-5 정확도 | 파라미터 수 |
|------|-------------|-------------|-----------|
| ResNet50 (Baseline) | ~72.0% | ~91.0% | 23.7M |
| SE-ResNet50 (Ours) | ~76.5% | ~93.5% | 26.2M |
| SE-ResNet50 + CutMix (Ours) | ~78.5% | ~94.5% | 26.2M |

### 학습 곡선

![Performance Comparison](assets/performance_comparison.png)

## 아키텍처

```
Input (3x32x32)
  └── Conv1 (3x3, stride=1) + BN + ReLU
  └── Layer1: 3x SE-Bottleneck (64)
  └── Layer2: 4x SE-Bottleneck (128)
  └── Layer3: 6x SE-Bottleneck (256)
  └── Layer4: 3x SE-Bottleneck (512)
  └── AdaptiveAvgPool2d → FC (100 classes)
```

## 실행 방법

### 환경 설정

```bash
pip install -r requirements.txt
```

### 학습

```bash
python train.py
```

실행 시:
- CIFAR-100 데이터셋 자동 다운로드
- SE-ResNet50 + CutMix 증강으로 50 에포크 학습
- Learning Rate Warmup (5 에포크) 후 Cosine Annealing 적용
- 최고 성능 모델을 `best_model.pth`로 저장
- 학습 곡선을 `training_curves.png`로 생성

### 추론

```bash
python inference.py --image <이미지 경로> --checkpoint best_model.pth --top_k 5
```

### 설정

주요 하이퍼파라미터는 `train.py`에서 수정 가능합니다:

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `num_epochs` | 50 | 총 학습 에포크 |
| `batch_size` | 128 | 미니배치 크기 |
| `learning_rate` | 0.1 | 초기 학습률 |
| `cutmix_prob` | 0.5 | CutMix 적용 확률 |
| `cutmix_alpha` | 1.0 | CutMix Beta 분포 파라미터 |

### 프로젝트 구조

```
enhanced-resnet50-cv/
├── model.py          # SE-ResNet50 아키텍처 정의
├── train.py          # CutMix 포함 학습 파이프라인
├── utils.py          # CutMix 유틸리티 및 손실 함수
├── inference.py      # 단일 이미지 추론 스크립트
├── requirements.txt  # 의존성 패키지 목록
├── assets/           # 학습 시각화 결과
├── .gitignore        # Git 무시 규칙
├── README.md         # 영문 프로젝트 문서
└── README_KR.md      # 한국어 프로젝트 문서
```

## 참고 문헌

- [Squeeze-and-Excitation Networks (Hu et al., 2018)](https://arxiv.org/abs/1709.01507)
- [CutMix: Regularization Strategy to Train Strong Classifiers (Yun et al., 2019)](https://arxiv.org/abs/1905.04899)
- [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385)
