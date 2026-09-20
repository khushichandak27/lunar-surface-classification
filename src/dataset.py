"""
PyTorch Dataset definition for Lunar Surface Classification.
Handles image loading, physics rotation, relief calculation, and data augmentation.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .preprocessing import prepare_lunar_input

class LunarSurfaceDataset(Dataset):
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        image_dir: str,
        is_train: bool = False,
        mode: str = 'stacked'
    ):
        self.df = metadata_df.reset_index(drop=True)
        self.image_dir = image_dir
        self.is_train = is_train
        self.mode = mode # 'stacked', 'optical', or 'dual'

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_id = row['image_id']
        sun_azimuth = float(row['sun_azimuth_angle'])
        img_path = os.path.join(self.image_dir, img_id)
        
        # Load raw grayscale image
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
            
        # Physics rotation + relief tensor prep
        if self.mode == 'dual':
            x_opt, x_rel = prepare_lunar_input(img_gray, sun_azimuth, mode='dual')
            # Random horizontal flip during training
            if self.is_train and np.random.rand() > 0.5:
                x_opt = torch.flip(x_opt, [2])
                x_rel = torch.flip(x_rel, [2])
            inputs = (x_opt, x_rel)
        else:
            inputs = prepare_lunar_input(img_gray, sun_azimuth, mode=self.mode)
            if self.is_train and np.random.rand() > 0.5:
                inputs = torch.flip(inputs, [2])
                
        if 'label' in row:
            label = torch.tensor(int(row['label']), dtype=torch.long)
            return inputs, label, img_id
        return inputs, img_id
