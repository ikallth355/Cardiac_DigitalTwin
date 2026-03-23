import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class CamusPngDataset(Dataset):
    def __init__(self, data_dir, subgroup_file, transform=None):
        """
        Args:
            data_dir: Path to CAMUS_public folder
            subgroup_file: Name of the txt file (e.g., 'subgroup_training.txt')
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load patient IDs from the provided subgroup file
        list_path = os.path.join(data_dir, subgroup_file)
        with open(list_path, 'r') as f:
            self.patient_list = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.patient_list)

    def _get_spacing_from_cfg(self, cfg_path):
        """Extracts pixel spacing from the CAMUS .cfg metadata file."""
        spacing = [1.0, 1.0] # Default fallback
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                for line in f:
                    if "PixelSpacing" in line:
                        # Format: PixelSpacing: 0.308 0.308
                        parts = line.split(":")[-1].strip().split()
                        spacing = [float(p) for p in parts]
        return spacing

    def __getitem__(self, idx):
        p_id = self.patient_list[idx]
        
        # Define paths for Image, Mask, and Metadata (CFG)
        # Assumes images are in training_pngs or testing_pngs
        # Adjust folder name if your PNGs are all in one directory
        img_path = os.path.join(self.data_dir, "training_pngs", f"{p_id}_4CH_ED.png")
        mask_path = os.path.join(self.data_dir, "training_pngs", f"{p_id}_4CH_ED_gt.png")
        cfg_path = os.path.join(self.data_dir, "database_nifti", p_id, f"{p_id}_4CH_ED.cfg")

        # Load Image and Mask
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        
        # Load physical spacing for the Digital Twin
        spacing = self._get_spacing_from_cfg(cfg_path)

        if self.transform:
            image = self.transform(image)
        
        # Convert mask to LongTensor for CrossEntropyLoss (0-3 labels)
        mask = torch.from_numpy(np.array(mask)).long()

        return {
            "image": image,
            "mask": mask,
            "patient_id": p_id,
            "spacing": torch.tensor(spacing, dtype=torch.float32)
        }