import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from models.unet_model import UNet
from pathlib import Path

def create_presentation_visuals():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the AI Brain
    model = UNet(n_channels=1, n_classes=4).to(device)
    model.load_state_dict(torch.load("models/weights/unet_cardiac.pth", map_location=device))
    model.eval()

    # 2. Select a sample patient (e.g., patient0049)
    p_id = "patient0049"
    img_path = Path(f"data/CAMUS_public/training_pngs/{p_id}/ED.png")
    
    if not img_path.exists():
        print(f"Error: Could not find image at {img_path}")
        return

    # 3. Process the image
    raw_img = Image.open(img_path).convert("L").resize((256, 256))
    img_tensor = T.ToTensor()(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 4. Create a high-quality side-by-side plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Left: The Input
    axes[0].imshow(raw_img, cmap='gray')
    axes[0].set_title(f"Input: Raw Ultrasound ({p_id})", fontsize=14)
    axes[0].axis('off')

    # Middle: The AI's "Eyes" (The Mask)
    # We use 'jet' or 'tab10' to make the classes (1, 2, 3) pop out
    axes[1].imshow(pred, cmap='tab10')
    axes[1].set_title("AI Segmentation Mask", fontsize=14)
    axes[1].axis('off')

    # Right: The Digital Twin Overlay
    axes[2].imshow(raw_img, cmap='gray')
    # Alpha 0.4 makes the color semi-transparent so you can see the heart underneath
    axes[2].imshow(pred, alpha=0.4, cmap='jet')
    axes[2].set_title("Digital Twin: Anatomical Alignment", fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    save_path = "digital_twin_visual_result.png"
    plt.savefig(save_path, dpi=300) # High resolution for slides
    print(f"✅ Presentation visual saved as: {save_path}")

if __name__ == "__main__":
    create_presentation_visuals()