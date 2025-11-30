# Python 3.10 + CUDA 환경
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# 작업 디렉토리
WORKDIR /app

# 시스템 패키지 설치 (OpenCV 의존성)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 전체 소스 복사
COPY . .

# Python 경로 설정 (ai_model을 import할 수 있게)
ENV PYTHONPATH=/app

# 기본 명령어 (docker-compose에서 override)
CMD ["python", "--version"]
