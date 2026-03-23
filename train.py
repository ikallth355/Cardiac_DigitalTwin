import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.unet_model import UNet
from src.data_loader import CamusPngDataset
import torchvision.transforms as T
from pathlib import Path
import os

# 1. Setup Device (Optimized for Windows Laptops)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Training System Initialized ---")
print(f"Device: {device}")

# 2. Hyperparameters
EPOCHS = 20
BATCH_SIZE = 4  # Kept small for laptop memory safety
LR = 1e-4
DATA_DIR = Path("data/CAMUS_public")

# 3. Smart Data Detection (Prevents FileNotFoundError)
def get_available_patients(base_path):
    existing_patients = []
    # Check both folders you mentioned
    for folder in ["training_pngs", "testing_pngs"]:
        dir_path = base_path / folder
        if dir_path.exists():
            # Finds folders named 'patientXXXX'
            found = [d.name for d in dir_path.iterdir() if d.is_dir() and d.name.startswith("patient")]
            existing_patients.extend(found)
    
    # Clean and sort the list
    unique_patients = sorted(list(set(existing_patients)))
    
    # Save to a new verified subgroup file
    verified_file = "subgroup_available.txt"
    with open(base_path / verified_file, "w") as f:
        for p in unique_patients:
            f.write(f"{p}\n")
    
    return verified_file, len(unique_patients)

# Run detection
verified_subgroup, patient_count = get_available_patients(DATA_DIR)
print(f"Detected {patient_count} patients on disk. Generated {verified_subgroup}")

# 4. Data Preparation
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

train_ds = CamusPngDataset(str(DATA_DIR), verified_subgroup, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# 5. Model, Loss, Optimizer
model = UNet(n_channels=1, n_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 6. Training Loop
print(f"\nStarting Training for {EPOCHS} Epochs...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for batch in train_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] -> Loss: {avg_loss:.4f}")

# 7. Save Weights
os.makedirs("models/weights", exist_ok=True)
weights_path = "models/weights/unet_cardiac.pth"
torch.save(model.state_dict(), weights_path)

print(f"\n--- Training Complete ---")
print(f"Weights saved successfully to: {weights_path}")