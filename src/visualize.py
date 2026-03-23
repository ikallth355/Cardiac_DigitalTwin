import matplotlib.pyplot as plt
import numpy as np

def save_twin_visualization(image, mask, p_id, output_folder="outputs/reports/"):
    plt.figure(figsize=(10, 5))
    
    # Original Ultrasound
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Patient {p_id}: Ultrasound")
    plt.axis('off')
    
    # AI Segmentation Overlay
    plt.subplot(1, 2, 2)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.imshow(mask, alpha=0.4, cmap='jet') # Overlay mask with transparency
    plt.title("AI Predicted 'Geometric State'")
    plt.axis('off')
    
    plt.savefig(f"{output_folder}/{p_id}_visual.png")
    plt.close()