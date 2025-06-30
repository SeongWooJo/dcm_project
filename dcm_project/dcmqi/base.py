import os
import shutil
import pydicom
import SimpleITK as sitk
from typing import List
import numpy as np
import dcm_project.read.filter as filter

def move_input(dicom_files : List[pydicom.Dataset], nii_image : sitk.Image, temp_dir : str) -> None:
    """
    DICOM 파일과 NIfTI 이미지를 임시 디렉토리로 이동합니다.

    Args:
        dicom_files (List[pydicom.Dataset]): DICOM 객체 리스트
        nii_image (sitk.Image): NIfTI 이미지 객체
        temp_dir (str): 임시 디렉토리 경로
    """
    temp_dicom_dir = os.path.join(temp_dir, 'dcm_series')
    os.makedirs(temp_dicom_dir, exist_ok=True)

    input_dicom_files = filter.sanitize_filter(dicom_files)
    for idx, ds in enumerate(input_dicom_files, start=1):
        filename = f"{idx:05d}.dcm"
        save_path = os.path.join(temp_dicom_dir, filename)
        ds.save_as(save_path)

    sitk.WriteImage(nii_image, os.path.join(temp_dir, 'seg.nii.gz'))
    print(f"[+] DICOM files and NIfTI image moved to {temp_dicom_dir}")

    unique_list = np.unique(nii_image)
    if len(unique_list) > 2:
        return "/data/metadata.json"
    elif len(unique_list) <= 2:
        if 1 in unique_list:
            return "/data/metadata1.json"
        elif 2 in unique_list:
            return "/data/metadata2.json"
    

def move_output(output_path : str):
    shutil.copy('/data/output/output.dcm', output_path)