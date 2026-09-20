"""
Physics-based preprocessing and feature engineering for lunar surface images.
- Sun-azimuth illumination normalization (counter-clockwise rotation by -azimuth)
- Illumination-invariant topographic relief extraction (Sobel & LoG)
"""

import cv2
import numpy as np
import torch

def rotate_to_standard_illumination(image: np.ndarray, sun_azimuth: float) -> np.ndarray:
    """
    Normalizes the illumination angle by rotating the lunar crop counter-clockwise
    by -sun_azimuth_angle. Edge replication prevents black border artifacts.
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_angle = -float(sun_azimuth)
    rot_mat = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated

def extract_topographic_relief(image_gray: np.ndarray) -> np.ndarray:
    """
    Extracts high-contrast topographic surface relief gradients using
    normalized Sobel edge magnitude. Serves as an illumination-invariant
    representation of surface elevation changes.
    """
    if image_gray.dtype != np.uint8:
        image_gray = np.clip(image_gray, 0, 255).astype(np.uint8)
        
    gx = cv2.Sobel(image_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    
    # Normalize to [0, 255]
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)

def prepare_lunar_input(
    image_gray: np.ndarray,
    sun_azimuth: float,
    mode: str = 'stacked'
) -> tuple:
    """
    Preprocesses a raw lunar grayscale crop:
    1. Sun-azimuth rotation
    2. Relief gradient extraction
    3. Normalization to [0, 1] range
    
    Modes:
    - 'optical': 3-channel grayscale [I, I, I]
    - 'stacked': 3-channel stacked relief [I, E, I]
    - 'dual': tuple of (optical_tensor, relief_tensor)
    """
    # 1. Physics rotation
    rotated = rotate_to_standard_illumination(image_gray, sun_azimuth)
    
    # 2. Topographic relief
    relief = extract_topographic_relief(rotated)
    
    # Convert to float32 [0, 1]
    rot_f = rotated.astype(np.float32) / 255.0
    rel_f = relief.astype(np.float32) / 255.0
    
    if mode == 'optical':
        tensor = np.stack([rot_f, rot_f, rot_f], axis=0) # (3, H, W)
        return torch.from_numpy(tensor).float()
    elif mode == 'stacked':
        tensor = np.stack([rot_f, rel_f, rot_f], axis=0) # (3, H, W)
        return torch.from_numpy(tensor).float()
    elif mode == 'dual':
        t_opt = torch.from_numpy(np.stack([rot_f, rot_f, rot_f], axis=0)).float()
        t_rel = torch.from_numpy(np.stack([rel_f, rel_f, rel_f], axis=0)).float()
        return t_opt, t_rel
    else:
        raise ValueError(f"Unknown mode: {mode}")
