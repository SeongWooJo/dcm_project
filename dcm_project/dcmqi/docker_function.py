import os
import shutil
import subprocess

def run_docker_command(metadata_file, output_path):
    # 4. Docker Compose 실행
    subprocess.run(["docker", "compose", "up", "-d"], check=True)

    # 5. 컨테이너 내부 명령 실행
    seg_cmd = (
        "docker exec dcmqi_container "
        "itkimage2segimage "
        "--inputImageList /data/seg.nii.gz "
        "--inputDICOMDirectory /data/dcm_series "
        f"--outputDICOM /data/output/output.dcm "
        f"--inputMetadata {metadata_file}"
    )
    subprocess.run(seg_cmd, shell=True, capture_output=True, text=True, check=True)
    shutil.copy('/data/output/output.dcm', output_path)