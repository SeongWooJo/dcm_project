import numpy as np, pydicom, copy
import SimpleITK as sitk
from typing import List
from scipy.ndimage import binary_erosion
from pydicom.uid import generate_uid

def edge_mask(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """채워진 마스크 ➜ 외곽선(1-pixel) 마스크"""
    if thickness < 1:
        return mask.astype(bool)
    eroded = binary_erosion(mask, iterations=thickness, border_value=0)
    return mask & ~eroded

def maybe_flip(seg_slice: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """ImageOrientationPatient에 맞춰 X/Y축 반전"""
    iop = list(map(float, ds.ImageOrientationPatient))
    row_x, row_y = iop[0], iop[1]     # row 방향 첫 두 성분
    col_x, col_y = iop[3], iop[4]     # column 방향
    out = seg_slice
    if row_x < 0:      # 좌우 뒤집힘
        out = np.fliplr(out)
    if col_y < 0:      # 상하 뒤집힘
        out = np.flipud(out)
    return out

def _packbits(mask: np.ndarray) -> bytes:
    """Overlay Plane용 비트 패킹 (Little-Endian, 16-bit word align)."""
    rows, cols = mask.shape
    pad = (16 - (cols % 16)) % 16
    if pad:
        mask = np.pad(mask, ((0, 0), (0, pad)), constant_values=0)
    # numpy ≥1.17: bitorder="little" 로 픽셀0→LSB 저장
    packed = np.packbits(mask.astype(np.uint8),
                         axis=1,
                         bitorder="little")
    return packed.tobytes()

def _add_plane(ds: pydicom.Dataset, mask: np.ndarray, plane_idx: int, desc: str):
    g = 0x6000 + plane_idx*2
    ds.add_new((g,0x0010),"US",mask.shape[0])
    ds.add_new((g,0x0011),"US",mask.shape[1])
    ds.add_new((g,0x0022),"LO",desc)
    ds.add_new((g,0x0040),"CS","G")
    ds.add_new((g,0x0050),"SS",[1,1])
    ds.add_new((g,0x0100),"US",1)
    ds.add_new((g,0x0102),"US",0)
    ds.add_new((g,0x3000),"OW",_packbits(mask))

def convert_seg_to_overlay_plane(
    dicom_files: List[pydicom.Dataset],
    nii_image: sitk.Image,
    outline: bool = True,            # ← True: 외곽선만
    edge_px: 1 = 1                  # 외곽선 두께
) -> List[pydicom.Dataset]:
    seg = sitk.GetArrayFromImage(nii_image)  # (Z,Y,X)
    if seg.shape[0] != len(dicom_files):
        raise ValueError("slice 수 불일치")
    # Z-정렬
    dcm_sorted = sorted(dicom_files,key=lambda d: float(d.ImagePositionPatient[2]))
    series_uid = generate_uid()

    out_ds = []

    # ### 디버깅용
    # edge_nii_path: str = "seg_edges.nii.gz"   # ★ 저장 경로
    # edge_vol = np.zeros_like(seg, dtype=np.uint8)
    # ###
    for z, (ds_src, seg_slice) in enumerate(zip(dcm_sorted, seg)):
        ds = copy.deepcopy(ds_src)
        #seg_slice = maybe_flip(seg_slice, ds)

        m1 = edge_mask(seg_slice==1, 1)
        #m2 = edge_mask(seg_slice==2, 1)

        _add_plane(ds, m1, plane_idx=0, desc="Tumor")
        #_add_plane(ds, m2, plane_idx=1, desc="Kidney")

        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID  = generate_uid()
        ds.SeriesDescription = "CT + Overlay (Tumor/Kidney)"
        ds.ImageType = ["DERIVED","PRIMARY","OVERLAY"]

        out_ds.append(ds)

        # edge_vol[z][m1] = 1
        # edge_vol[z][m2] = 2
    # # --------- NIfTI(Edge) 저장 -----------
    # edge_img = sitk.GetImageFromArray(edge_vol)
    # # 원본 공간 정보 복제
    # edge_img.SetOrigin(nii_image.GetOrigin())
    # edge_img.SetSpacing(nii_image.GetSpacing())
    # edge_img.SetDirection(nii_image.GetDirection())
    # sitk.WriteImage(edge_img, edge_nii_path)
    # print(f"[✓] 윤곽선 NIfTI 저장 완료 → {edge_nii_path}")
    return out_ds



def convert_seg_to_overlay_plane2(
    dicom_files: List[pydicom.Dataset],
    nii_image: sitk.Image
) -> List[pydicom.Dataset]:
    seg_array = sitk.GetArrayFromImage(nii_image)  # shape: (Z, Y, X)
    assert seg_array.shape[0] == len(dicom_files), "Mismatch between DICOM and segmentation slices."

    result_files = []
    dcm_sorted = sorted(dicom_files,key=lambda d: float(d.ImagePositionPatient[2]))
    for i, ds in enumerate(dcm_sorted):
        ds = copy.deepcopy(ds)
        seg_slice = seg_array[i]
        hu_slice = ds.pixel_array.astype(np.int16)
        
        # Normalize HU (-1024~1024) → [0, 255] Grayscale
        hu_clip = np.clip(hu_slice, -1024, 1024)
        hu_norm = ((hu_clip + 1024) / 2048.0 * 255.0).astype(np.uint8)
        rgb = np.stack([hu_norm] * 3, axis=-1)

        # label 1·2 → 외곽선 or 채워진 마스크
        m1 = edge_mask(seg_slice==1, 1)
        m2 = edge_mask(seg_slice==2, 1)

        # Color overlay
        rgb[m1 == 1] = [255, 0, 0]     # 빨강
        rgb[m2 == 2] = [0, 255, 0]     # 초록

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