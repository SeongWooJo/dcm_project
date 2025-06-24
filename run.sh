#!/bin/bash

FOLDER_NAME=$(basename "$PWD")
IMAGE_NAME="$FOLDER_NAME"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CONTAINER_NAME="${FOLDER_NAME}_${TIMESTAMP}"

MODE=${1:-dev}  # 기본값: dev

# 실행 모드 (dev/code/prod)
MODE=${1:-dev}

# 운영체제 구분
UNAME=$(uname -s)

if [[ "$UNAME" == MINGW* || "$UNAME" == CYGWIN* || "$UNAME" == MSYS* ]]; then
    # Windows (Git Bash 등)
    HOST_DIR=$(pwd -W)  # Windows 절대경로로 변환
    echo "[*] Windows 환경 감지됨 → HOST_DIR=${HOST_DIR}"
else
    # Linux / WSL / macOS
    HOST_DIR=$PWD
    echo "[*] Linux/macOS 환경 감지됨 → HOST_DIR=${HOST_DIR}"
fi

# Dockerfile 경로 결정
if [ "$MODE" = "dev" ]; then
    DOCKERFILE="Dockerfile.dev"
elif [ "$MODE" = "code" ]; then
    DOCKERFILE="Dockerfile.code"
else
    DOCKERFILE="Dockerfile"  # 기본 Dockerfile (prod)
fi

# Docker 이미지가 없으면 빌드 (모드별 Dockerfile 사용)
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "[+] Building Docker image '$IMAGE_NAME' with $DOCKERFILE ..."
    docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" .
fi

# 컨테이너 실행 (모드별)
if [ "$MODE" = "dev" ]; then
    echo "[*] 개발모드: 전체 디렉토리 마운트"
    docker run -it --rm \
        --name "$CONTAINER_NAME" \
        -v "$HOST_DIR":/app \
        "$IMAGE_NAME"

elif [ "$MODE" = "code" ]; then
    echo "[*] 운영모드: 코드 디렉토리만 마운트"
    docker run -it --rm \
        --name "$CONTAINER_NAME" \
        -v "$HOST_DIR/$FOLDER_NAME":/app/$FOLDER_NAME \
        "$IMAGE_NAME"

else
    echo "[*] 배포모드: 마운트 없이 실행"
    docker run -it --rm \
        --name "$CONTAINER_NAME" \
        "$IMAGE_NAME"
fi