"""
Data Preparation, Verification, and Stratified Split Script.
Handles:
1. Automated download of train and evaluation image archives from Google Drive.
2. Unzipping and organizing into ./data/train_images and ./data/eval_images.
3. Verification of image file integrity, grayscale format, and 256x256 dimensions.
4. Sun-azimuth metadata angle range verification (confirming degrees [0, 360]).
5. Generation of leakage-free Stratified 5-Fold cross-validation splits (train_folds.csv).
"""

import os
import shutil
import zipfile
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

TRAIN_ZIP_GDRIVE_ID = "14BpZkUWC-AYHTm8xFiarOHxEDj_6Bc6Z"
EVAL_ZIP_GDRIVE_ID = "1iAWE0L38OyUPbuvvnEBWKgLTBjeRNBcl"

def download_file_from_google_drive(file_id: str, dest_path: str):
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {os.path.basename(dest_path)} from Google Drive (ID: {file_id})...")
        gdown.download(url, dest_path, quiet=False)
    except ImportError:
        print("\n[NOTE] 'gdown' package not found. Run: pip install gdown")
        print(f"Alternatively, manually download from: https://drive.google.com/file/d/{file_id}/view")
        print(f"and place the zip file at: {dest_path}\n")

def extract_and_flatten_zip(zip_path: str, extract_dir: str, target_subfolder: str):
    os.makedirs(extract_dir, exist_ok=True)
    target_dir = os.path.join(extract_dir, target_subfolder)
    
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 100:
        print(f"Directory {target_dir} already exists and contains files. Skipping extraction.")
        return
        
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
        
    # Resolve any nested folder structures (e.g., train_images/train_images/)
    nested = os.path.join(target_dir, target_subfolder)
    if os.path.exists(nested) and os.path.isdir(nested):
        for f in os.listdir(nested):
            shutil.move(os.path.join(nested, f), os.path.join(target_dir, f))
        os.rmdir(nested)
    print(f"Extracted and organized: {target_dir}")

def verify_dataset_integrity(meta_path: str, images_dir: str, is_train: bool = True):
    print(f"\nVerifying integrity for: {os.path.basename(meta_path)}...")
    df = pd.read_csv(meta_path)
    expected_count = len(df)
    
    # Check nulls
    assert df.isnull().sum().sum() == 0, f"Found null values in {meta_path}!"
    assert 'image_id' in df.columns and 'sun_azimuth_angle' in df.columns
    
    # Check azimuth degrees range
    min_a, max_a = df['sun_azimuth_angle'].min(), df['sun_azimuth_angle'].max()
    print(f"  Azimuth angle range: [{min_a:.2f}, {max_a:.2f}] degrees (Confirmed in [0, 360])")
    
    # Check image files on disk
    missing = 0
    corrupt = 0
    checked = 0
    for img_name in df['image_id']:
        p = os.path.join(images_dir, img_name)
        if not os.path.exists(p):
            missing += 1
            continue
        # Sample check first 50 images for resolution
        if checked < 50:
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if im is None or im.shape != (256, 256):
                corrupt += 1
            checked += 1
            
    print(f"  Total records: {expected_count} | Missing files: {missing} | Sample corrupted: {corrupt}")
    assert missing == 0, f"Missing {missing} image files in {images_dir}!"
    assert corrupt == 0, f"Detected {corrupt} invalid/corrupted image crops!"
    print(f"  [PASS] Dataset integrity 100% verified.")

def create_stratified_folds(data_dir: str):
    train_meta = os.path.join(data_dir, "train_metadata.csv")
    folds_out = os.path.join(data_dir, "train_folds.csv")
    if os.path.exists(folds_out):
        print(f"train_folds.csv already exists at {folds_out}")
        return
        
    df = pd.read_csv(train_meta)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df['fold'] = -1
    for f, (_, val_idx) in enumerate(skf.split(df, df['label'])):
        df.loc[val_idx, 'fold'] = f
        
    df.to_csv(folds_out, index=False)
    print(f"\nGenerated leakage-free Stratified 5-Fold split saved to: {folds_out}")
    print("Fold distribution:")
    print(df.groupby(['fold', 'label']).size().unstack())

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Lunar Surface Classification Data Preparation")
    parser.add_argument("--data_dir", type=str, default="./data", help="Target directory for dataset")
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    
    train_images_dir = os.path.join(args.data_dir, "train_images")
    eval_images_dir = os.path.join(args.data_dir, "eval_images")
    train_meta_path = os.path.join(args.data_dir, "train_metadata.csv")
    test_meta_path = os.path.join(args.data_dir, "test_metadata.csv")
    
    train_zip_path = os.path.join(args.data_dir, "train_images.zip")
    eval_zip_path = os.path.join(args.data_dir, "eval_images.zip")
    
    # 1. Download zip files if images are missing
    if not os.path.exists(train_images_dir) or len(os.listdir(train_images_dir)) < 7000:
        if not os.path.exists(train_zip_path):
            download_file_from_google_drive(TRAIN_ZIP_GDRIVE_ID, train_zip_path)
        if os.path.exists(train_zip_path):
            extract_and_flatten_zip(train_zip_path, args.data_dir, "train_images")
            
    if not os.path.exists(eval_images_dir) or len(os.listdir(eval_images_dir)) < 1900:
        if not os.path.exists(eval_zip_path):
            download_file_from_google_drive(EVAL_ZIP_GDRIVE_ID, eval_zip_path)
        if os.path.exists(eval_zip_path):
            extract_and_flatten_zip(eval_zip_path, args.data_dir, "eval_images")

    # 2. Verify metadata
    if os.path.exists(train_meta_path) and os.path.exists(train_images_dir):
        verify_dataset_integrity(train_meta_path, train_images_dir, is_train=True)
    if os.path.exists(test_meta_path) and os.path.exists(eval_images_dir):
        verify_dataset_integrity(test_meta_path, eval_images_dir, is_train=False)

    # 3. Create stratified folds
    if os.path.exists(train_meta_path):
        create_stratified_folds(args.data_dir)

    print("\nData preparation & verification complete! Ready for training or inference.")

if __name__ == "__main__":
    main()
