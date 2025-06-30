    #!/bin/bash

    # 경로를 저장한 텍스트 파일
    SAVE_META_FILE="save_metadata"  # 예: 파일 내용이 "/data/metadata.json"

    # 내용 읽어서 변수에 저장 (줄바꿈 제거)
    META_FILE=$(cat "$SAVE_META_FILE" | tr -d '\r\n')

    # 유효성 검사
    if [ -z "$META_FILE" ]; then
    echo "[ERROR] 메타데이터 경로가 비어있습니다."
    exit 1
    fi

    echo "[INFO] 사용할 메타데이터 경로: $META_FILE"

    docker-compose up

    # Docker 명령 실행
    docker exec dcmqi_container \
    itkimage2segimage \
    --inputImageList /data/seg.nii.gz \
    --inputDICOMDirectory /data/dcm_series \
    --outputDICOM /data/output/output.dcm \
    --inputMetadata "$META_FILE"
