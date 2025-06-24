from dcm_project.read import base, filter
from dcm_project.analysis import physical_data
from dcm_project.dcmqi.base import move_input
from dcm_project.dcmqi.docker_function import run_docker_command

import pydicom
import os
import SimpleITK as sitk

if __name__ == '__main__':
    dicom_dir = 'input/dcm_input/'
    nii_dir = 'input/nii_input/'
    temp_dir = 'dcm_project/dcmqi/'
    case_number = 1

    dicom_path = os.path.join(dicom_dir, str(case_number).zfill(3))
    nii_path = os.path.join(nii_dir, 'case_' + str(case_number).zfill(5) + '.nii.gz')
    dicom_files = base.read_dicom_files(dicom_path)
    print(f"Total DICOM files found: {len(dicom_files)}")

    n_phase_files = filter.n_phase_filter(dicom_files, image_type='Axial')
    result = base.print_unique_series(n_phase_files)

    image = sitk.ReadImage(nii_path)
    image_uint8 = sitk.Cast(image, sitk.sitkUInt8)
    np_image_uint8 = sitk.GetArrayFromImage(image_uint8)

    dicom_origin = physical_data.get_origin_with_dcm(n_phase_files, image_uint8)
    image_uint8.SetOrigin(dicom_origin)

    #### 라이브러리 실행용
    meta_path = move_input(n_phase_files, image_uint8, temp_dir)
    output_path = f'output/seg_{str(case_number).zfill(3)}.nii.gz'
    run_docker_command(meta_path, output_path)

    

