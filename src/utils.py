import nibabel as nib
import numpy as np
from typing import Tuple, Dict

def load_nifti(file_path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Loads a NIfTI file and returns the data array and the pixel spacing.
    """
    img = nib.load(file_path)
    data = img.get_fdata()
    # Spacing is (x_res, y_res, slice_thickness) in mm
    spacing = img.header.get_zooms()
    return data, spacing

def extract_metadata(p_id: str, file_path: str) -> Dict:
    """
    Extracts essential patient metadata for the Digital Twin profile.
    """
    img = nib.load(file_path)
    header = img.header
    return {
        "patient_id": p_id,
        "resolution": list(header.get_zooms()[:2]), # x and y resolution in mm
        "unit": "mm"
    }

def calculate_volume_simpson(mask: np.ndarray, spacing: Tuple[float, float], label: int = 1) -> float:
    """
    Estimates Left Ventricle (LV) volume using the Method of Disks (Simpson's Rule).
    
    Formula: Volume = (Area_of_slices * height_of_slices)
    In 2D ultrasound (CAMUS), we approximate this by identifying the 
    longitudinal axis and dividing the LV into disks.
    
    Args:
        mask: 2D Segmentation mask.
        spacing: (pixel_width, pixel_height) in mm.
        label: The integer value for the LV in the mask (usually 1).
        
    Returns:
        Volume in milliliters (mL).
    """
    # 1. Isolate the LV mask
    lv_mask = (mask == label).astype(np.uint8)
    
    # 2. Find the vertical extent of the LV (Long Axis)
    rows = np.any(lv_mask, axis=1)
    if not np.any(rows):
        return 0.0
    
    ymin, ymax = np.where(rows)[0][[0, -1]]
    height_mm = (ymax - ymin) * spacing[1]
    
    # 3. Divide into disks (slices)
    # We sum pixels in each row to find the diameter of each disk
    disk_diameters_px = np.sum(lv_mask[ymin:ymax, :], axis=1)
    disk_diameters_mm = disk_diameters_px * spacing[0]
    
    # 4. Sum volume of disks: V = sum( (pi * r^2) * slice_height )
    # where r = diameter / 2
    slice_height_mm = spacing[1]
    disk_volumes_mm3 = np.pi * (disk_diameters_mm / 2)**2 * slice_height_mm
    
    total_volume_mm3 = np.sum(disk_volumes_mm3)
    
    # Convert mm^3 to mL (1000 mm^3 = 1 mL)
    return total_volume_mm3 / 1000.0

def classify_ef(ef_value: float) -> str:
    """
    Clinical classification based on Ejection Fraction percentage.
    """
    if ef_value >= 55:
        return "Normal"
    elif 45 <= ef_value < 55:
        return "Mildly Reduced"
    elif 30 <= ef_value < 45:
        return "Moderately Reduced"
    else:
        return "Severely Reduced"