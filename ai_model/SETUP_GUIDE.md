# 🚀 무인 상점 도난 탐지 시스템 설치 가이드

## 📋 설치 순서

### 1. Conda 환경 생성
```bash
conda env create -f environment.yml
conda activate unmaned_shoplifting
```

### 2. PyTorch 설치 (GPU 지원)
**NVIDIA GPU가 있는 경우 (권장):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU만 사용하는 경우:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. 설치 확인
```bash
python check_gpu.py
```

## 🔧 시스템 요구사항

- **Python**: 3.11+
- **CUDA**: 11.8+ (GPU 사용 시)
- **RAM**: 8GB 이상 권장
- **GPU**: NVIDIA GPU (선택사항, 성능 향상)

## 📝 주요 패키지 버전

- **PyTorch**: 2.7.1+cu118 (GPU) / 2.7.1+cpu (CPU)
- **OpenCV**: 4.12.0.88
- **Ultralytics**: 8.3.203 (YOLO)
- **Polars**: 1.33.1 (데이터 처리)

## ⚠️ 문제 해결

### GPU 인식 안 됨
1. NVIDIA 드라이버 최신 버전 설치
2. CUDA Toolkit 설치 확인
3. `nvidia-smi` 명령어로 GPU 상태 확인

### 패키지 설치 오류
1. Conda 환경이 활성화되었는지 확인
2. 인터넷 연결 상태 확인
3. PyTorch 인덱스 URL이 정확한지 확인

## 🎯 배포 환경 고려사항

- **안정성**: 안정 버전의 PyTorch 사용 (nightly 버전 지양)
- **호환성**: 다양한 GPU 환경에서 작동하도록 fallback 시스템 구현
- **이식성**: CPU 모드로도 실행 가능하도록 설계

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. 위 설치 순서를 정확히 따랐는지
2. `python check_gpu.py` 실행 결과
3. 오류 메시지 전문