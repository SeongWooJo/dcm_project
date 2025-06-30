from dcm_project.read.base import check_orientation, read_dicom_tag
import pydicom
import os
from typing import List
import SimpleITK as sitk
import numpy as np
import copy

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

def sanitize_filter(dicom_files: List[pydicom.Dataset]) -> List[pydicom.Dataset]:
    """
    익명화된 DICOM 파일에서 필수 태그들을 유효한 값으로 보정하여 반환.

    Args:
        dicom_list (List[pydicom.Dataset]): 수정할 DICOM 객체 리스트

    Returns:
        List[pydicom.Dataset]: 수정된 DICOM 객체 리스트
    """
    result_files = []

    for ds in dicom_files:
        ds = ds.copy()

        # ✅ 환자 정보 비우기
        ds.PatientID = ""
        ds.PatientName = ""
        ds.PatientBirthDate = ""
        ds.PatientSex = ""
        ds.PatientAge = ""

        # ✅ 날짜/시간 정보 비우기
        ds.StudyDate = ""
        ds.SeriesDate = ""
        ds.AcquisitionDate = ""
        ds.ContentDate = ""
        ds.StudyTime = ""
        ds.SeriesTime = ""
        ds.AcquisitionTime = ""
        ds.ContentTime = ""

        # ✅ 장비 관련 날짜 정보도 비우기 (존재할 경우만)
        if "DateOfLastCalibration" in ds:
            ds.DateOfLastCalibration = ""
        if "TimeOfLastCalibration" in ds:
            ds.TimeOfLastCalibration = ""

        result_files.append(ds)

    return result_files

def change_name_filter(dicom_files: List[pydicom.Dataset], change_name : str) -> List[pydicom.Dataset]:
    result_files = []
    
    for ds in dicom_files:
        ds = ds.copy()  # 원본을 보호하기 위해 deepcopy 대신 pydicom 방식으로 shallow copy
        
        # 이름 관련 필드 변경
        ds.PatientName = change_name
        ds.StudyDescription = change_name
        ds.SeriesDescription = change_name
        
        result_files.append(ds)

    return result_files


def change_pixel_data(dicom_list: List[pydicom.Dataset], nii_image: sitk.Image) -> List[pydicom.Dataset]:
    # NIfTI 이미지를 numpy 배열로 변환
    nii_array = sitk.GetArrayFromImage(nii_image)  # (z, y, x) 형태

    # z 방향 슬라이스 수 확인
    if len(dicom_list) != nii_array.shape[0]:
        raise ValueError(f"NIfTI 슬라이스 수 ({nii_array.shape[0]})와 DICOM 수 ({len(dicom_list)})가 일치하지 않습니다.")

    # Slice 위치 기준으로 정렬 (Z-axis 방향)
    dicom_list_sorted = sorted(dicom_list, key=lambda d: float(d.ImagePositionPatient[2]))

    # DICOM 리스트 복사 및 픽셀 데이터 수정
    updated_dicom_list = []
    for idx, (ds, new_slice) in enumerate(zip(dicom_list_sorted, nii_array)):
        ds = ds.copy()
        mapped = np.zeros_like(new_slice, dtype=np.int16)
        mapped[new_slice == 1] = 1124
        mapped[new_slice == 2] = 1524
        ds.PixelData = mapped.tobytes()
        ds.Rows, ds.Columns = mapped.shape
        
        # 필요한 경우 UID도 새로 생성 가능
        # ds.SOPInstanceUID = pydicom.uid.generate_uid()

        updated_dicom_list.append(ds)

    return updated_dicom_list


def create_overlay_rgb(ct_array: np.ndarray, seg_array: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    # 1. 정규화된 CT grayscale → 3채널로 확장
    norm_ct = ((ct_array - np.min(ct_array)) / (np.max(ct_array) - np.min(ct_array)) * 255).astype(np.uint8)
    rgb = np.stack([norm_ct] * 3, axis=-1).astype(np.float32)  # shape: (H, W, 3)

    # 2. 라벨별 오버레이 색 정의
    color_map = {
        1: np.array([255, 0, 0], dtype=np.float32),   # 빨강
        2: np.array([0, 255, 0], dtype=np.float32),   # 초록
    }

    # 3. alpha blending 적용
    for label_value, color in color_map.items():
        mask = seg_array == label_value
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color

    # 4. 최종 출력: uint8 형 변환
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def make_secondary_capture_dicom(template_ds: pydicom.Dataset, rgb_image: np.ndarray) -> pydicom.Dataset:
    ds = template_ds.copy()
    ds.file_meta = ds.file_meta.copy()

    # SC 형식 메타 정보로 설정
    ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    ds.Modality = "SC"
    ds.PhotometricInterpretation = "RGB"
    ds.SamplesPerPixel = 3
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0

    ds.Rows, ds.Columns = rgb_image.shape[0], rgb_image.shape[1]
    ds.PixelData = rgb_image.tobytes()

    # optional: 인스턴스 UID 재설정
    ds.SOPInstanceUID = pydicom.uid.generate_uid()

    return ds

def convert_seg_to_secondary_capture(
    dicom_files: List[pydicom.Dataset],
    nii_image: sitk.Image
) -> List[pydicom.Dataset]:
    seg_array = sitk.GetArrayFromImage(nii_image)  # (z, y, x)
    dicom_sorted = sorted(dicom_files, key=lambda d: float(d.ImagePositionPatient[2]))
    if seg_array.shape[0] != len(dicom_sorted):
        raise ValueError("NIfTI 슬라이스 수와 DICOM 슬라이스 수가 일치하지 않습니다.")

    result_files = []

    for ds, seg_slice in zip(dicom_sorted, seg_array):
        ct_array = ds.pixel_array.astype(np.float32)

        rgb_image = create_overlay_rgb(ct_array, seg_slice)
        new_ds = make_secondary_capture_dicom(ds, rgb_image)
        result_files.append(new_ds)

    return result_files


def convert_seg_to_secondary_capture2(
    dicom_files: List[pydicom.Dataset],
    nii_image: sitk.Image
) -> List[pydicom.Dataset]:
    seg_array = sitk.GetArrayFromImage(nii_image)  # shape: (Z, Y, X)
    assert seg_array.shape[0] == len(dicom_files), "Mismatch between DICOM and segmentation slices."

    result_files = []

    for i, ds in enumerate(dicom_files):
        ds = copy.deepcopy(ds)
        seg_slice = seg_array[i]
        hu_slice = ds.pixel_array.astype(np.int16)

        origin = nii_image.GetOrigin()           # (x0, y0, z0)
        spacing = nii_image.GetSpacing()         # (sx, sy, sz)
        size = nii_image.GetSize()               # (nx, ny, nz)
        direction = nii_image.GetDirection()     # 3x3 flatten (row-major)

        # direction matrix
        dir_matrix = [
            direction[0:3],
            direction[3:6],
            direction[6:9]
        ]
        seg_data = sitk.GetArrayFromImage(nii_image)  # shape: (z, y, x)
        if seg_data.shape[0] != len(dicom_files):
            raise ValueError(f"NIfTI slices ({seg_data.shape[0]}) ≠ DICOM slices ({len(dicom_files)})")
        
        ### flip 여부 판단
        dicom_z_positions = [float(ds.ImagePositionPatient[2]) for ds in dicom_files]

        # 3. z 방향 비교
        is_dicom_increasing = dicom_z_positions[-1] > dicom_z_positions[0]  # True면 아래→위 방향
        is_seg_increasing = origin[2] < (origin[2] + spacing[2] * (size[2] - 1))

        # 4. 방향 다르면 flip
        if is_dicom_increasing != is_seg_increasing:
            print("[INFO] Z축 방향이 달라 segmentation을 뒤집습니다.")
            seg_data = np.flip(seg_data, axis=0)
        else:
            print("[INFO] Z축 방향이 일치합니다. Flip 불필요.")
        
        # Normalize HU (-1024~1024) → [0, 255] Grayscale
        hu_clip = np.clip(hu_slice, -1024, 1024)
        hu_norm = ((hu_clip + 1024) / 2048.0 * 255.0).astype(np.uint8)
        rgb = np.stack([hu_norm] * 3, axis=-1)

        # Color overlay
        rgb[seg_slice == 1] = [255, 0, 0]     # 빨강
        rgb[seg_slice == 2] = [0, 255, 0]     # 초록

        # DICOM 태그 수정 for RGB
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.Rows, ds.Columns = rgb.shape[:2]
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PlanarConfiguration = 0  # interleaved: R1G1B1, R2G2B2 ...

        ds.PixelData = rgb.tobytes()

        # SC 저장
        ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
        ds.Modality = "OT"
        ds.SeriesDescription = "Segmentation Overlay SC"
        ds.ImageType = ['DERIVED', 'SECONDARY']
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.SOPInstanceUID = pydicom.uid.generate_uid()

        result_files.append(ds)

    return result_files