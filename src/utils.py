import numpy as np

def calculate_volume_simpson(mask, spacing):
    """
    Calculates volume using a simplified Simpson's method.
    We specifically look for Class 1 (Left Ventricle).
    """
    # CRITICAL FIX: The model outputs labels 0, 1, 2, 3. 
    # Class 1 is the Left Ventricle.
    lv_mask = (mask == 1).astype(np.uint8)
    
    # If no LV pixels are found, return 0
    if np.sum(lv_mask) == 0:
        return 0.0

    pixel_area_mm2 = spacing[0] * spacing[1]
    
    # Count pixels per row (disk method approximation)
    row_counts = np.sum(lv_mask, axis=1)
    
    # Filter out rows with no LV pixels
    disk_areas = row_counts[row_counts > 0] * pixel_area_mm2
    
    # Volume = Sum of disk areas * height (1 pixel height in mm)
    # We use spacing[1] as the vertical height per pixel
    volume_mm3 = np.sum(disk_areas) * spacing[1]
    
    # Convert mm3 to mL (1000 mm3 = 1 mL)
    return volume_mm3 / 1000.0

def classify_ef(ef_pct):
    """Clinical classification based on Ejection Fraction."""
    if ef_pct >= 55: return "Normal"
    if 45 <= ef_pct < 55: return "Mildly Reduced"
    if 30 <= ef_pct < 45: return "Moderately Reduced"
    return "Severely Reduced"