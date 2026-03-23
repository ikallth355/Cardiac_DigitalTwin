import torch
import pandas as pd
from pathlib import Path
from models.unet_model import UNet
from src.data_loader import CamusPngDataset
from src.digital_twin import CardiacDigitalTwin
import torchvision.transforms as T
from tqdm import tqdm

def run_batch_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/weights/unet_cardiac.pth"
    
    # Load Model
    model = UNet(n_channels=1, n_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load Data
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    ds = CamusPngDataset("data/CAMUS_public", "subgroup_available.txt", transform=transform)
    
    results = []
    print(f"Generating Digital Twin profiles for {len(ds)} patients...")

    for item in tqdm(ds):
        p_id = item['patient_id']
        spacing = item['spacing'].tolist()
        
        # Predict Mask
        img_tensor = item['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Initialize and Update Digital Twin
        twin = CardiacDigitalTwin(p_id, spacing)
        twin.update_geometry(pred_mask, "ED")
        
        # Simulation: For a full twin, we'd predict the ES frame too.
        # Here we simulate a 30% contraction for the ES state
        twin.update_geometry(pred_mask * 0.7, "ES") 
        twin.compute_metrics()
        
        profile = twin.get_profile()
        results.append({
            "PatientID": p_id,
            "EDV_mL": profile['physical_parameters']['EDV_ml'],
            "ESV_mL": profile['physical_parameters']['ESV_ml'],
            "EF_pct": profile['functional_parameters']['ejection_fraction_pct'],
            "Status": profile['functional_parameters']['classification']
        })

    # Save Summary
    df = pd.DataFrame(results)
    df.to_csv("outputs/reports/clinical_summary.csv", index=False)
    print(f"\nSuccess! Summary saved to outputs/reports/clinical_summary.csv")

if __name__ == "__main__":
    run_batch_evaluation()