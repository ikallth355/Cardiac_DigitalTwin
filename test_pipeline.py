import torch
import numpy as np
from models.unet_model import UNet
from src.data_loader import CamusPngDataset
from src.digital_twin import CardiacDigitalTwin
import torchvision.transforms as T

def run_sanity_check():
    print("--- Phase 1: Model Architecture Check ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=1, n_classes=4).to(device)
    dummy_input = torch.randn(1, 1, 256, 256).to(device)
    output = model(dummy_input)
    print(f"Model Output Shape: {output.shape} (Expected: [1, 4, 256, 256])")

    print("\n--- Phase 2: Data Loader & Metadata Check ---")
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    # Note: Ensure 'subgroup_training.txt' exists in your data folder
    try:
        ds = CamusPngDataset("data/CAMUS_public", "subgroup_testing.txt", transform=transform)
        sample = ds[0]
        print(f"Patient ID: {sample['patient_id']}")
        print(f"Image Tensor Shape: {sample['image'].shape}")
        print(f"Pixel Spacing: {sample['spacing'].tolist()} mm")
    except Exception as e:
        print(f"Data Loader Error: {e}")
        return

    print("\n--- Phase 3: Digital Twin Logic Check ---")
    # We simulate a "perfect" mask for the LV (a circle in the middle)
    dummy_mask = np.zeros((256, 256))
    dummy_mask[100:150, 100:150] = 1 # Label 1 is Left Ventricle
    
    # Initialize Twin with the patient's real spacing
    twin = CardiacDigitalTwin(sample['patient_id'], sample['spacing'].tolist())
    
    # Simulate ED and ES updates
    twin.update_geometry(dummy_mask, "ED")
    twin.update_geometry(dummy_mask * 0.8, "ES") # Slightly smaller for Systole
    
    twin.compute_metrics()
    profile = twin.get_profile()
    
    print(f"Estimated EDV: {profile['physical_parameters']['EDV_ml']} mL")
    print(f"Ejection Fraction: {profile['functional_parameters']['ejection_fraction_pct']}%")
    print(f"Clinical Status: {profile['functional_parameters']['classification']}")

if __name__ == "__main__":
    run_sanity_check()