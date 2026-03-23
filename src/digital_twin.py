import numpy as np
from src.utils import calculate_volume_simpson, classify_ef

class CardiacDigitalTwin:
    def __init__(self, patient_id: str, spacing: tuple):
        """
        Initializes the Digital Twin 'Profile' for a specific patient.
        """
        self.profile = {
            "metadata": {
                "patient_id": patient_id,
                "spatial_resolution_mm": list(spacing),
                "status": "initialized"
            },
            "geometric_state": {
                "ED_mask": None,
                "ES_mask": None
            },
            "physical_parameters": {
                "EDV_ml": 0.0,  # End-Diastolic Volume
                "ESV_ml": 0.0   # End-Systolic Volume
            },
            "functional_parameters": {
                "stroke_volume_ml": 0.0,
                "ejection_fraction_pct": 0.0,
                "classification": "Unknown"
            }
        }

    def update_geometry(self, mask: np.ndarray, phase: str):
        """
        Updates the twin with new 'sensor data' (segmentation masks).
        phase: 'ED' or 'ES'
        """
        if phase.upper() not in ['ED', 'ES']:
            raise ValueError("Phase must be 'ED' or 'ES'")
            
        self.profile["geometric_state"][f"{phase.upper()}_mask"] = mask
        
        # Calculate volume immediately when geometry is updated
        spacing = self.profile["metadata"]["spatial_resolution_mm"]
        volume = calculate_volume_simpson(mask, spacing)
        self.profile["physical_parameters"][f"{phase.upper()}V_ml"] = round(volume, 2)

    def compute_metrics(self):
        """
        The 'Brain' of the twin: Calculates EF and clinical status.
        """
        edv = self.profile["physical_parameters"]["EDV_ml"]
        esv = self.profile["physical_parameters"]["ESV_ml"]
        
        if edv > 0:
            # Stroke Volume = EDV - ESV
            sv = edv - esv
            # EF = (SV / EDV) * 100
            ef = (sv / edv) * 100
            
            self.profile["functional_parameters"].update({
                "stroke_volume_ml": round(sv, 2),
                "ejection_fraction_pct": round(ef, 2),
                "classification": classify_ef(ef)
            })
            self.profile["metadata"]["status"] = "computed"
        else:
            print(f"Warning: EDV is 0 for patient {self.profile['metadata']['patient_id']}")

    def get_profile(self):
        """Returns the full Digital Twin data object."""
        return self.profile