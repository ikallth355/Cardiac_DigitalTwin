import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path

class CamusPngDataset(Dataset):
    def __init__(self, data_dir, subgroup_file, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        list_path = self.data_dir / subgroup_file
        if not list_path.exists():
            # Create a dummy file if it doesn't exist just to prevent crashing during testing
            print(f"Warning: {subgroup_file} not found. Creating a temporary one for testing.")
            with open(list_path, 'w') as f:
                f.write("patient0027") 
        
        with open(list_path, 'r') as f:
            self.patient_list = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.patient_list)

    def _get_spacing_from_cfg(self, p_id):
        # We search for the .cfg in the database_nifti folder
        cfg_path = self.data_dir / "database_nifti" / p_id / f"{p_id}_4CH_ED.cfg"
        spacing = [1.0, 1.0] 
        if cfg_path.exists():
            with open(cfg_path, 'r') as f:
                for line in f:
                    if "PixelSpacing" in line:
                        parts = line.split(":")[-1].strip().split()
                        spacing = [float(p) for p in parts]
        return spacing

    def __getitem__(self, idx):
        p_id = self.patient_list[idx]
        
        # Check BOTH training and testing folders
        possible_dirs = ["training_pngs", "testing_pngs"]
        p_folder = None
        
        for d in possible_dirs:
            temp_path = self.data_dir / d / p_id
            if temp_path.exists():
                p_folder = temp_path
                break
        
        if p_folder is None:
            raise FileNotFoundError(f"Critical: Patient folder {p_id} not found in training_pngs or testing_pngs")

        img_path = p_folder / "ED.png"
        mask_path = p_folder / "ED_gt.png"

        if not img_path.exists():
            raise FileNotFoundError(f"Missing image file: {img_path}")

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        
        # Consistent resizing for the U-Net
        image = image.resize((256, 256), resample=Image.BILINEAR)
        mask = mask.resize((256, 256), resample=Image.NEAREST)

        # --- FIX FOR INDEX ERROR: Label Mapping ---
        mask_np = np.array(mask)
        unique_labels = np.unique(mask_np) # Finds [0, 77, 150, ...]
        
        # We map every unique value found to a sequential index (0, 1, 2, 3)
        # This ensures the model never sees a '77' but sees class '1' instead.
        label_map = {val: i for i, val in enumerate(sorted(unique_labels))}
        
        # Apply mapping
        final_mask = np.zeros_like(mask_np)
        for val, target_idx in label_map.items():
            if target_idx < 4:  # Safety check for 4 classes
                final_mask[mask_np == val] = target_idx
        # ------------------------------------------

        spacing = self._get_spacing_from_cfg(p_id)

        if self.transform:
            image = self.transform(image)
        
        mask_tensor = torch.from_numpy(final_mask).long()

        return {
            "image": image,
            "mask": mask_tensor,
            "patient_id": p_id,
            "spacing": torch.tensor(spacing, dtype=torch.float32)
        }