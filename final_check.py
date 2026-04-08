import torch
import numpy as np
import torchvision.transforms as T
from models.unet_model import UNet
from src.data_loader import CamusPngDataset

print("--- Diagnostic Tool Started ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=1, n_classes=4).to(device)

# DIRECT PATH - No 'os' needed
weights_path = "models/weights/unet_cardiac.pth"

try:
    # Loading the brain
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"SUCCESS: Loaded weights from {weights_path}")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load {weights_path}")
    print(f"Reason: {e}")
    print("Check if the file exists in models/weights/ folder!")

model.eval()

# Load just 1 image
transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
ds = CamusPngDataset("data/CAMUS_public", "subgroup_available.txt", transform=transform)
item = ds[0]

with torch.no_grad():
    image = item['image'].unsqueeze(0).to(device)
    output = model(image)
    pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

print(f"Patient ID: {item['patient_id']}")
unique_vals = np.unique(pred)
print(f"Unique pixel values predicted: {unique_vals}")

bg_count = np.sum(pred == 0)
lv_count = np.sum(pred == 1)

print(f"Background pixels (0): {bg_count}")
print(f"Left Ventricle (1): {lv_count}")

if bg_count > 1000 and lv_count > 100:
    print("\n--- STATUS: THE DIGITAL TWIN IS ALIVE! ---")
else:
    print("\n--- STATUS: STILL CALIBRATING (Check if loss is dropping) ---")