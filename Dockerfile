# Python 3.10 기반 슬림 이미지
FROM python:3.10-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 설치
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 코드 복사 (선택)
COPY . .

# 기본 실행 명령 (필요시)
CMD ["/bin/bash"]