import torch
import numpy as np
import json
from pathlib import Path
from models.unet_model import UNet
from src.data_loader import CamusPngDataset
from src.digital_twin import CardiacDigitalTwin
import torchvision.transforms as T

def evaluate_patient(patient_id, model_path="models/weights/unet_cardiac.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the Trained Brain (U-Net)
    model = UNet(n_channels=1, n_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Setup Data
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    ds = CamusPngDataset("data/CAMUS_public", "subgroup_testing.txt", transform=transform)
    
    # Find the specific patient in the dataset
    patient_data = None
    for item in ds:
        if item['patient_id'] == patient_id:
            patient_data = item
            break
    
    if not patient_data:
        print(f"Error: Patient {patient_id} not found.")
        return

    # 3. Predict the "Sensor Data" (Segmentation Masks)
    # Note: In a real scenario, we'd predict both ED and ES images. 
    # For this demo, we'll use the ground truth or the model's prediction.
    image_tensor = patient_data['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 4. Update the Digital Twin
    spacing = patient_data['spacing'].tolist()
    twin = CardiacDigitalTwin(patient_id, spacing)
    
    # Update with predicted geometry (Simulating ED/ES for the profile)
    twin.update_geometry(pred_mask, "ED")
    # In a full run, you'd repeat the prediction for the ES frame here
    twin.update_geometry(pred_mask * 0.7, "ES") # Dummy contraction for testing
    
    twin.compute_metrics()
    
    # 5. Output the Profile
    profile = twin.get_profile()
    
    # Save to outputs/reports/
    output_path = Path(f"outputs/reports/{patient_id}_twin.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(profile, f, indent=4)
        
    print(f"--- Digital Twin Profile Generated for {patient_id} ---")
    print(f"EF: {profile['functional_parameters']['ejection_fraction_pct']}%")
    print(f"Status: {profile['functional_parameters']['classification']}")
    print(f"Report saved to: {output_path}")

if __name__ == "__main__":
    # Test it on one of your patients
    evaluate_patient("patient0027")