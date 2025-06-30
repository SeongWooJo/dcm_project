from highdicom.seg import Segmentation, SegmentDescription
from highdicom.content import AlgorithmIdentificationSequence
from highdicom.sr.coding import CodedConcept
from highdicom.seg.enum import SegmentAlgorithmTypeValues
from highdicom.content import PlaneOrientationSequence, PlanePositionSequence, PixelMeasuresSequence
from datetime import datetime
import pydicom
import SimpleITK as sitk
import os
import numpy as np

import dcm_project.read.filter as filter
from typing import List

def create_dicom_seg(dicom_files: List[pydicom.Dataset], sitk_seg: sitk.Image, output_path):
    origin = sitk_seg.GetOrigin()           # (x0, y0, z0)
    spacing = sitk_seg.GetSpacing()         # (sx, sy, sz)
    size = sitk_seg.GetSize()               # (nx, ny, nz)
    direction = sitk_seg.GetDirection()     # 3x3 flatten (row-major)

    # direction matrix
    dir_matrix = [
        direction[0:3],
        direction[3:6],
        direction[6:9]
    ]
    seg_data = sitk.GetArrayFromImage(sitk_seg)  # shape: (z, y, x)
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

    # 4. Build PixelMeasures, PlaneOrientation, PlanePositions
    pixel_measures = PixelMeasuresSequence(
        pixel_spacing=(spacing[1], spacing[0]),       # y, x
        slice_thickness=spacing[2]
    )

    image_orientation = [
        direction[0], direction[1], direction[2],  # row direction
        direction[3], direction[4], direction[5],  # column direction
    ]

    plane_orientation = PlaneOrientationSequence(
        coordinate_system="PATIENT",
        image_orientation=image_orientation
    )

    plane_positions = []
    z_indices = range(size[2]) if is_dicom_increasing else reversed(range(size[2]))

    for k in z_indices:
        # spacing[2] × 방향벡터(dir_matrix[2]) × 인덱스(k)
        offset = [
            origin[i] + spacing[2] * k * (1 if is_dicom_increasing else -1)
            for i in range(3)
        ]
        plane_positions.append(
            PlanePositionSequence(
                coordinate_system="PATIENT",
                image_position=offset
            )
        )
    # 3. Build Segment description
    algorithm_id = AlgorithmIdentificationSequence(
        name="MySegTool",
        family=CodedConcept(
            value="113037",
            scheme_designator="DCM",
            meaning="Segmentation algorithm"
        ),
        version="1.0",
        source="MyOrganization",
        parameters={"threshold": "0.5", "model": "UNet"}
    )   
    segment_descriptions = [
        SegmentDescription(
            segment_number=1,
            segment_label='Tumor',
            segmented_property_category=CodedConcept("M-8000/3", "SCT", "Morphologically Abnormal Structure"),  # CID 7150
            segmented_property_type=CodedConcept("M-8000/3", "SCT", "Neoplasm"),  # CID 7151
            algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=algorithm_id
        ),
        SegmentDescription(
            segment_number=2,
            segment_label='Kidney',
            segmented_property_category=CodedConcept("T-D0050", "SRT", "Tissue"),  # CID 7150
            segmented_property_type=CodedConcept("T-71000", "SRT", "Kidney"),     # CID 7151
            algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=algorithm_id
        )
    ]

    seg = Segmentation(
        source_images=dicom_files,
        pixel_array=seg_data,              # (2*Z, Y, X)
        segmentation_type="LABELMAP",                  # ✅ 핵심 변경점
        segment_descriptions=segment_descriptions,
        series_instance_uid=pydicom.uid.generate_uid(),
        series_number=300,
        sop_instance_uid=pydicom.uid.generate_uid(),
        instance_number=1,
        manufacturer="YourOrganization",
        manufacturer_model_name="YourSegmentationModel",
        software_versions="1.0.0",
        device_serial_number="SN-001",
        pixel_measures=pixel_measures,
        plane_orientation=plane_orientation,
        plane_positions=plane_positions,         # ✅ frame 수 = Z * 2
        omit_empty_frames=False
    )
    seg.SeriesDescription = "Segmentation"
    # 5. Save to DICOM SEG file
    pydicom.filewriter.dcmwrite(output_path, seg, write_like_original=False)
    print(f"[✓] DICOM SEG saved to: {output_path}")