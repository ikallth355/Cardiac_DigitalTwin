import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.unet_model import UNet
from src.data_loader import CamusPngDataset
import torchvision.transforms as T
import os

# 1. Setup Device (Windows Laptop check)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# 2. Hyperparameters
EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-4
DATA_DIR = "data/CAMUS_public"

# 3. Data Prep
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

train_ds = CamusPngDataset(DATA_DIR, "subgroup_training.txt", transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# 4. Model, Loss, Optimizer
model = UNet(n_channels=1, n_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 5. Training Loop
print("Starting Training...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for images, masks, _ in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(train_loader):.4f}")

# 6. Save Weights
torch.save(model.state_state_dict(), "models/weights/unet_cardiac.pth")
print("Model saved to models/weights/unet_cardiac.pth")