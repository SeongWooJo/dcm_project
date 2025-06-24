from dcm_project.read.base import check_orientation, read_dicom_tag
import pydicom
import os
from typing import List

def n_phase_filter(dicom_list: List[pydicom.Dataset], image_type: str = 'Axial') -> List[pydicom.Dataset]:
    """
    DICOM 파일 리스트에서 특정 이미지 타입에 해당하는 시리즈만 필터링하여 반환.

    Args:
        dicom_list (List[pydicom.Dataset]): DICOM 객체 리스트
        image_type (str): 필터링할 이미지 타입 (기본값: 'Axial')

    Returns:
        List[pydicom.Dataset]: 필터링된 DICOM 객체 리스트
    """
    filtered_list = []
    input_dict = {"SeriesNumber": "N/A", "ContrastBolusAgent": "N/A", "SeriesDescription": "N/A", "ImageOrientationPatient": None}
    
    for ds in dicom_list:
        orientation_type = check_orientation(ds)
        tags = read_dicom_tag(ds, input_dict)
        if orientation_type == image_type and ("pre" in tags["SeriesDescription"].lower() or tags["ContrastBolusAgent"] == "N/A"):
            filtered_list.append(ds)

    return filtered_list

def sanitize_filter(dicom_list: List[pydicom.Dataset]) -> List[pydicom.Dataset]:
    """
    익명화된 DICOM 파일에서 필수 태그들을 유효한 값으로 보정하여 반환.

    Args:
        dicom_list (List[pydicom.Dataset]): 수정할 DICOM 객체 리스트

    Returns:
        List[pydicom.Dataset]: 수정된 DICOM 객체 리스트
    """
    for ds in dicom_list:
        # 환자 정보
        ds.PatientID = ""
        ds.PatientName = ""
        ds.PatientBirthDate = ""
        ds.PatientSex = ""

    return dicom_list
