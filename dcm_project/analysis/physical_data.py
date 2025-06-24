import SimpleITK as sitk
import pydicom
import numpy as np
from typing import List


def get_origin_with_dcm(dicom_files : List[pydicom.Dataset], nii_image : sitk.Image) -> bool:
    
    image_uint8 = sitk.Cast(nii_image, sitk.sitkUInt8)

    if len(dicom_files) > 1:
        dicom_z_positions = [float(ds.ImagePositionPatient[2]) for ds in dicom_files]
        is_dicom_increasing = dicom_z_positions[-1] > dicom_z_positions[0]  # True면 아래→위 방향
    
        origin = image_uint8.GetOrigin()
        spacing = image_uint8.GetSpacing()
        size = image_uint8.GetSize()

        is_seg_increasing = origin[2] < (origin[2] + spacing[2] * (size[2] - 1))
        if is_dicom_increasing == is_seg_increasing:
            dicom_origin = [float(x) for x in dicom_files[0].ImagePositionPatient]
        else:
            dicom_origin = [float(x) for x in dicom_files[-1].ImagePositionPatient]
    else:
        dicom_origin = [float(x) for x in dicom_files[0].ImagePositionPatient]
    
    return dicom_origin