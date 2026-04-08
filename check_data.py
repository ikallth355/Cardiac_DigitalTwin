import numpy as np
from PIL import Image
import os

# Check the first patient's ground truth
mask_path = "data/CAMUS_public/training_pngs/patient0049/ED_gt.png"
mask = np.array(Image.open(mask_path))
print(f"Unique values in raw mask: {np.unique(mask)}")