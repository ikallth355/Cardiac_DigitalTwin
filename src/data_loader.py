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
            print(f"Warning: {subgroup_file} not found. Creating a temporary one.")
            with open(list_path, 'w') as f:
                f.write("patient0027") 
        
        with open(list_path, 'r') as f:
            self.patient_list = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.patient_list)

    def _get_spacing_from_cfg(self, p_id):
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
        
        # Locate patient folder
        possible_dirs = ["training_pngs", "testing_pngs"]
        p_folder = None
        for d in possible_dirs:
            temp_path = self.data_dir / d / p_id
            if temp_path.exists():
                p_folder = temp_path
                break
        
        if p_folder is None:
            raise FileNotFoundError(f"Critical: Patient {p_id} folder missing.")

        img_path = p_folder / "ED.png"
        mask_path = p_folder / "ED_gt.png"

        # 1. Load Image and Mask
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        
        # 2. Resize - Use NEAREST for mask to avoid creating more "fake" gray values
        image = image.resize((256, 256), resample=Image.BILINEAR)
        mask = mask.resize((256, 256), resample=Image.NEAREST)

        mask_np = np.array(mask)
        
        # 3. --- THE "FUZZY MASK" RECOVERY LOGIC ---
        # Instead of exact matches, we use threshold ranges to capture blurred edges
        final_mask = np.zeros_like(mask_np, dtype=np.int64)
        
        # Capture Left Ventricle (LV) - Target ~77
        final_mask[(mask_np >= 40) & (mask_np < 110)] = 1
        
        # Capture Myocardium (MYO) - Target ~150
        final_mask[(mask_np >= 110) & (mask_np < 200)] = 2
        
        # Capture Left Atrium (LA) - Target ~255
        final_mask[mask_np >= 200] = 3
        # ------------------------------------------

        spacing = self._get_spacing_from_cfg(p_id)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)
        
        mask_tensor = torch.from_numpy(final_mask).long()

        return {
            "image": image,
            "mask": mask_tensor,
            "patient_id": p_id,
            "spacing": torch.tensor(spacing, dtype=torch.float32)
        }