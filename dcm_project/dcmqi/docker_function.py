import os
import shutil
import subprocess

save_meta_path = 'save_metadata/'

def load_variable_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()
    
def run_docker_command(metadata_file, output_path):
    # subprocess.run(["docker-compose", "up", "-d"], check=True)
    seg_cmd = (
        "docker exec dcmqi_container "
        "itkimage2segimage "
        "--inputImageList /data/seg.nii.gz "
        "--inputDICOMDirectory /data/dcm_series "
        "--outputDICOM /data/output/output.dcm "
        "--inputMetadata " + metadata_file
    )
    subprocess.run(seg_cmd, shell=True, capture_output=True, text=True, check=True)
    #shutil.copy('/data/output/output.dcm', output_path)

if __name__ == '__main__':
    meta_path = load_variable_from_file(save_meta_path)
    output_path = 'output.dcm'

    run_docker_command(meta_path, output_path)