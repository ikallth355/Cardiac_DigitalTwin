import numpy as np
from src.utils import calculate_volume_simpson, classify_ef

class CardiacDigitalTwin:
    def __init__(self, patient_id: str, spacing: tuple):
        self.profile = {
            "metadata": {
                "patient_id": patient_id,
                "spatial_resolution_mm": list(spacing),
                "status": "initialized"
            },
            "geometric_state": {"ED_mask": None, "ES_mask": None},
            "physical_parameters": {"EDV_ml": 0.0, "ESV_ml": 0.0},
            "functional_parameters": {
                "stroke_volume_ml": 0.0,
                "ejection_fraction_pct": 0.0,
                "classification": "Unknown"
            }
        }

    def update_geometry(self, mask: np.ndarray, phase: str):
        if phase.upper() not in ['ED', 'ES']:
            raise ValueError("Phase must be 'ED' or 'ES'")
            
        self.profile["geometric_state"][f"{phase.upper()}_mask"] = mask
        
        # Pull resolution
        spacing = self.profile["metadata"]["spatial_resolution_mm"]
        
        # Calculate volume
        volume = calculate_volume_simpson(mask, spacing)
        self.profile["physical_parameters"][f"{phase.upper()}V_ml"] = round(volume, 2)

    def compute_metrics(self):
        edv = self.profile["physical_parameters"]["EDV_ml"]
        esv = self.profile["physical_parameters"]["ESV_ml"]
        
        if edv > 0:
            sv = edv - esv
            ef = (sv / edv) * 100
            
            self.profile["functional_parameters"].update({
                "stroke_volume_ml": round(sv, 2),
                "ejection_fraction_pct": round(max(0, ef), 2), # EF can't be negative
                "classification": classify_ef(ef)
            })
            self.profile["metadata"]["status"] = "computed"
        else:
            self.profile["functional_parameters"]["classification"] = "Detection Failed"

    def get_profile(self):
        return self.profile