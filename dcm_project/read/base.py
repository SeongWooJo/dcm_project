import pydicom
import numpy as np
import os
from typing import List

def read_dicom_files(dicom_dir : str) -> List[pydicom.Dataset]:
    dicom_files = []
    for root, _, files in os.walk(dicom_dir):
        for file in files:
            if file.lower().endswith(".dcm"):
                path = os.path.join(root, file)
                try:
                    ds = pydicom.dcmread(path)
                    dicom_files.append(ds)
                except Exception as e:
                    print(f"[!] Failed to read {file}: {e}")
    dicom_files = sorted(dicom_files, key=lambda x: (int(x.SeriesNumber), int(x.InstanceNumber)))
    return dicom_files

def read_dicom_tag(dicom_file : pydicom.dataset.FileDataset, tag_dict : dict) -> dict:
    """
    Reads specific DICOM tags from a DICOM file.

    Parameters:
    dicom_file (str): DICOM file.
    tag_dict (dict): Dictionary containing DICOM tags to read with default option.
    Example: {'SeriesNumber' : 'N/A', 'ContrastBolusAgent' : 'N/A'}
    Returns:
    dict: Dictionary with tag names as keys and their values.
    """
    result = {}
    #print(dicom_file)
    for tag_name, tag_value in tag_dict.items():
        result[tag_name] = getattr(dicom_file, tag_name, tag_value)

    return result

def check_orientation(dicom_file : pydicom.dataset.FileDataset) -> str:

    input_dict = {'ImageOrientationPatient' : None}
    result = read_dicom_tag(dicom_file, input_dict)
    raw_axis = result['ImageOrientationPatient']

    axis = None
    if raw_axis is None or len(raw_axis) != 6:
        axis = "Unknown"
    else:
        row = np.array(raw_axis[:3])
        col = np.array(raw_axis[3:])
        normal = np.cross(row, col)
        # 절댓값이 가장 큰 축에 따라 판단
        abs_normal = np.abs(normal)

        if abs_normal[2] > 1 - 1e-4:  # Z 축
            axis = "Axial"
        elif abs_normal[1] > 1 - 1e-4:  # Y 축
            axis = "Coronal"
        elif abs_normal[0] > 1 - 1e-4:  # X 축
            axis = "Sagittal"
        else:
            axis = "Oblique"
    
    return axis

def print_unique_series(dicom_files : List[pydicom.Dataset]) -> None:
    unique_series = set()
    input_dict = {"SeriesNumber": "N/A", "ContrastBolusAgent": "N/A", "SeriesDescription": "N/A", "ImageOrientationPatient": None}
    
    for ds in dicom_files:
        tags = read_dicom_tag(ds, input_dict)
        orientation_type = check_orientation(ds)
        unique_series.add((tags["SeriesNumber"], tags["SeriesDescription"], tags["ContrastBolusAgent"], orientation_type))
    
    print("\n[📋 Unique Series List]")
    for sn, desc, ce, axis in sorted(unique_series):
        print(f"Series Number: {sn}, Description: {desc}, Contrast: {ce}, Image : {axis}")
