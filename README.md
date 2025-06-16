# Safety AI - Federated Learning

9클래스 의료 이미지 분류를 위한 Federated Learning 프로젝트입니다.

##  **지원하는 FL 전략**

- **FedAvg**: 기본 연합학습 알고리즘
- **FedProx**: Proximal term을 추가한 안정성 향상 방법

##  **프로젝트 구조**

```
Safety_AI/
├── train/                    # 핵심 FL 구현
│   ├── federated.py         # FL 클라이언트 및 메인 로직
│   ├── strategies.py        # FL 전략 구현 (FedAvg, FedBN, FedProx)
│   ├── models.py           # 모델 초기화 (ResNet, EfficientNet)
│   └── loader.py           # 데이터 로더 유틸리티
├── config/
│   ├── fl/                 # FL 훈련 설정
│   │   ├── fedavg.yaml
│   │   ├── fedbn.yaml
│   │   └── fedprox.yaml
│   └── split/              # 데이터 분할 설정
│       ├── dirichlet_alpha1.yaml    # 강한 non-IID
│       ├── dirichlet_alpha5.yaml    # 중간 non-IID
│       ├── dirichlet_alpha10.yaml   # 약한 non-IID
│       └── quasi_iid.yaml           # 거의 IID
├── scripts/
│   └── dataset_split.py    # 데이터셋 분할 스크립트
├── data/
│   ├── train/raw/          # 훈련 데이터 (9클래스)
│   ├── test/               # 테스트 데이터 (제공됨)
│   └── split/              # 생성된 스플릿 JSON 파일들
└── run_federated.py        # 메인 실행 스크립트
```

## **사용법**

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 데이터셋 스플릿 생성

훈련 데이터를 `data/train/raw/`에 준비한 후:

```bash
# non-IID 분할 (alpha=1.0, 강한 클래스 집중)
python scripts/dataset_split.py --split config/split/dirichlet_alpha1.yaml

# 중간 non-IID 분할 (alpha=5.0)
python scripts/dataset_split.py --split config/split/dirichlet_alpha5.yaml

# 약한 non-IID 분할 (alpha=10.0)
python scripts/dataset_split.py --split config/split/dirichlet_alpha10.yaml

# 거의 IID 분할 (alpha=100.0)
python scripts/dataset_split.py --split config/split/quasi_iid.yaml
```

스플릿을 생성하면 `data/split/` 폴더에 JSON 파일과 각 클라이언트의 클래스 분포를
확인할 수 있는 플랏(`*_dist.png`)이 저장됩니다. JSON에는 클라이언트별 엔트로피도
포함되므로 데이터가 얼마나 non-IID한지 정량적으로 확인할 수 있습니다.

### 3. Federated Learning 실행

```bash
# FedAvg 실행
python run_federated.py --config config/fl/fedavg.yaml

# FedBN 실행 (BatchNorm 로컬 유지)
python run_federated.py --config config/fl/fedbn.yaml

# FedProx 실행 (Proximal term)
python run_federated.py --config config/fl/fedprox.yaml
```

### 4. 중앙 집중식 베이스라인 실행

```bash
python run_centralized.py --config config/centralized/custom9.yaml
```

## **데이터 분할 방식**

Dirichlet 분포를 사용하여 non-IID 데이터 분할:

- **α = 1.0**: 매우 강한 non-IID (각 클라이언트가 소수 클래스에 집중)
- **α = 5.0**: 중간 정도 non-IID 
- **α = 10.0**: 약한 non-IID
- **α = 100.0**: 거의 IID (균등 분배)

## **주요 기능**

### **올바른 FedBN 구현**
- 서버: BatchNorm 파라미터를 집계에서 제외
- 클라이언트: 로컬 BatchNorm 통계 유지
- 비-BN 파라미터만 연합 평균화

### **FedProx Proximal Term**
- μ(mu) 파라미터로 제어
- 로컬 업데이트를 글로벌 모델에 근접하게 유지
- 클라이언트 이질성 완화

### **자동 GPU 지원**
- CUDA 사용 가능 시 자동으로 GPU 활용
- CPU fallback 지원

### **유연한 모델 지원**
- ResNet50, ResNet34
- EfficientNet-B4, EfficientNet-B0
- 자동 이미지 크기 조정

##  **설정 옵션**

### 모델 설정
```yaml
model:
  name: resnet50          # resnet50, resnet34, efficientnet_b4, efficientnet_b0
  output_dim: 9           # 클래스 수
```

### 훈련 설정
```yaml
train:
  strategy: fedavg        # fedavg, fedbn, fedprox
  rounds: 50              # FL 라운드 수
  local_epochs: 5         # 로컬 에포크 수
  batch_size: 32
  lr: 0.001
  mu: 0.1                 # FedProx 전용
```

### FL 설정
```yaml
fl:
  min_fit_clients: 5      # 최소 참여 클라이언트
  min_available_clients: 5 # 최소 가용 클라이언트
  fraction_fit: 1.0       # 참여 비율
```

##  **주의사항**

1. **데이터 준비**: `data/train/raw/`에 9클래스 훈련 데이터 필요
2. **스플릿 선행**: FL 실행 전 반드시 데이터셋 스플릿 생성
3. **메모리**: GPU 메모리에 따라 batch_size 조정 필요
4. **테스트 데이터**: `data/test/` 폴더가 자동으로 사용됨

## **개선된 코드 품질**

- ✅ **올바른 FedBN 구현**: BN 파라미터 로컬 유지
- ✅ **완전한 FedProx**: Proximal term 적용
- ✅ **견고한 에러 처리**: 파일 존재성 검증
- ✅ **명확한 설정 분리**: 전략별 독립적 설정
- ✅ **자동 디바이스 감지**: GPU/CPU 자동 선택
- ✅ **최신 Flower 지원**: `client_fn`이 Context 객체를 받아 호환성 향상

## **Docker 사용 예시**

```bash
docker build -t safety-ai .
docker run --gpus all -it -v $(pwd):/workspace -v $(pwd)/data:/workspace/data safety-ai

# Powershell
docker run --gpus all -it `
  --shm-size=6g `
  -v "$(Get-Location):/workspace" `
  -v "$(Get-Location)/data:/workspace/data" `
  safety-ai
```
