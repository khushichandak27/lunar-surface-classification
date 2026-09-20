"""
Ensemble Inference and Submission Generation Script.
Implements:
1. Physics-based sun-azimuth rotation & topographic relief extraction for test crops.
2. Test-Time Augmentation (TTA: original + horizontal flip).
3. Probability fusion across all 7 model families with data-validated weights.
4. Optimal operating threshold (0.52) tuned via PR curve on 5-fold CV.
5. Strict validation of submission.csv (2,000 rows, header: image_id,label, 0 nulls, binary 0/1).
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

from src.dataset import LunarSurfaceDataset
from src.models import (
    DualBranchResNet,
    build_resnet18,
    build_resnet34,
    build_efficientnet_b0,
    build_efficientnet_b2,
    build_convnext_tiny
)

def predict_tta(model: nn.Module, loader: DataLoader, device: torch.device, is_dual: bool = False) -> np.ndarray:
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            if is_dual:
                (x_opt, x_rel), _ = batch
                x_opt = x_opt.to(device, non_blocking=True)
                x_rel = x_rel.to(device, non_blocking=True)
                with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    out_orig = torch.softmax(model(x_opt, x_rel), dim=1)[:, 1]
                    out_flip = torch.softmax(model(torch.flip(x_opt, [3]), torch.flip(x_rel, [3])), dim=1)[:, 1]
            else:
                inputs, _ = batch
                inputs = inputs.to(device, non_blocking=True)
                with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    out_orig = torch.softmax(model(inputs), dim=1)[:, 1]
                    out_flip = torch.softmax(model(torch.flip(inputs, [3])), dim=1)[:, 1]
            p_tta = (out_orig + out_flip) / 2.0
            all_probs.extend(p_tta.cpu().numpy())
    return np.array(all_probs)

def validate_submission_format(sub_df: pd.DataFrame, test_df: pd.DataFrame):
    print("\n" + "=" * 50)
    print("RUNNING STRICT SUBMISSION VALIDATION CHECKS")
    print("=" * 50)
    
    # Check 1: Row count
    assert len(sub_df) == len(test_df), f"[FAIL] Row count mismatch: expected {len(test_df)}, got {len(sub_df)}"
    print(f"[PASS] Row count check: exactly {len(sub_df)} prediction rows.")
    
    # Check 2: Column names
    assert list(sub_df.columns) == ['image_id', 'label'], f"[FAIL] Columns mismatch: {list(sub_df.columns)}"
    print("[PASS] Column names check: exactly ['image_id', 'label'].")
    
    # Check 3: Missing / Null values
    assert sub_df.isnull().sum().sum() == 0, "[FAIL] Found null / NaN values in submission!"
    print("[PASS] Null values check: 0 missing values.")
    
    # Check 4: Binary labels {0, 1}
    unique_labels = set(sub_df['label'].unique())
    assert unique_labels.issubset({0, 1}), f"[FAIL] Non-binary labels detected: {unique_labels}"
    print(f"[PASS] Label values check: only valid binary labels {unique_labels}.")
    
    # Check 5: Sequential alignment with test metadata
    assert (sub_df['image_id'].values == test_df['image_id'].values).all(), "[FAIL] Image IDs out of order!"
    print("[PASS] Image ID alignment check: 100% matched in exact sequential order.")
    
    c0 = (sub_df['label'] == 0).sum()
    c1 = (sub_df['label'] == 1).sum()
    print(f"\nFinal Class Breakdown:")
    print(f"  Class 0 (Depth): {c0} ({c0 / len(sub_df) * 100:.2f}%)")
    print(f"  Class 1 (Rise):  {c1} ({c1 / len(sub_df) * 100:.2f}%)")
    print("[SUCCESS] submission.csv is 100% competition-ready!\n")

def main():
    parser = argparse.ArgumentParser(description="Lunar Surface Classification Inference Pipeline")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing test_metadata.csv and eval_images")
    parser.add_argument("--models_dir", type=str, default="./models", help="Directory containing trained model checkpoints (.pt)")
    parser.add_argument("--output_csv", type=str, default="./submission.csv", help="Path to save the final submission.csv")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--threshold", type=float, default=0.52, help="Decision threshold for Class 1 (Rise)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    test_meta_path = os.path.join(args.data_dir, "test_metadata.csv")
    eval_images_dir = os.path.join(args.data_dir, "eval_images")
    if not os.path.exists(test_meta_path):
        raise FileNotFoundError(f"Missing {test_meta_path}. Place test_metadata.csv in {args.data_dir}")

    test_df = pd.read_csv(test_meta_path)
    print(f"Loaded {len(test_df)} test samples from {test_meta_path}")

    # Data loaders for optical, stacked relief, and dual-branch
    ds_optical = LunarSurfaceDataset(test_df, eval_images_dir, is_train=False, mode='optical')
    loader_optical = DataLoader(ds_optical, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ds_stacked = LunarSurfaceDataset(test_df, eval_images_dir, is_train=False, mode='stacked')
    loader_stacked = DataLoader(ds_stacked, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ds_dual = LunarSurfaceDataset(test_df, eval_images_dir, is_train=False, mode='dual')
    loader_dual = DataLoader(ds_dual, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_configs = [
        ('convnext_tiny_stacked', lambda: build_convnext_tiny(pretrained=False), loader_stacked, False, "convnext_tiny_stacked_relief_fold_*.pt", 0.25),
        ('effnet_b2_stacked', lambda: build_efficientnet_b2(pretrained=False), loader_stacked, False, "effnet_b2_stacked_relief_fold_*.pt", 0.15),
        ('resnet34_stacked', lambda: build_resnet34(pretrained=False), loader_stacked, False, "resnet34_stacked_relief_fold_*.pt", 0.15),
        ('dual_branch_resnet18', lambda: DualBranchResNet('resnet18', pretrained=False), loader_dual, True, "dual_branch_resnet18_fold_*.pt", 0.15),
        ('resnet18_stacked', lambda: build_resnet18(pretrained=False), loader_stacked, False, "resnet18_stacked_relief_fold_*.pt", 0.10),
        ('efficientnet_b0_optical', lambda: build_efficientnet_b0(pretrained=False), loader_optical, False, "efficientnet_b0_fold_*.pt", 0.10),
        ('resnet18_optical', lambda: build_resnet18(pretrained=False), loader_optical, False, "resnet18_fold_*.pt", 0.10)
    ]

    family_predictions = {}
    active_weights = {}

    for fam_name, builder_fn, loader, is_dual, pattern, weight in model_configs:
        ckpts = sorted(glob.glob(os.path.join(args.models_dir, pattern)))
        if not ckpts:
            print(f"Warning: No checkpoints found for {fam_name} with pattern '{pattern}' in {args.models_dir}. Skipping.")
            continue
            
        print(f"\nRunning inference for {fam_name} ({len(ckpts)} checkpoints found)...")
        model = builder_fn().to(device)
        fold_probs = []
        for ckpt in ckpts:
            print(f"  Loading {os.path.basename(ckpt)} + running TTA...")
            state = torch.load(ckpt, weights_only=True, map_location=device)
            model.load_state_dict(state)
            probs = predict_tta(model, loader, device, is_dual=is_dual)
            fold_probs.append(probs)
            
        family_predictions[fam_name] = np.mean(fold_probs, axis=0)
        active_weights[fam_name] = weight
        print(f"  -> {fam_name} mean P(Rise): {family_predictions[fam_name].mean():.4f}")

    if not family_predictions:
        raise RuntimeError("No model checkpoints were found! Please check --models_dir path.")

    # Normalize weights to active models
    total_w = sum(active_weights.values())
    norm_weights = {k: v / total_w for k, v in active_weights.items()}
    print("\n" + "=" * 50)
    print("ENSEMBLE WEIGHTS:")
    for k, w in norm_weights.items():
        print(f"  {k}: {w * 100:.1f}%")
    print("=" * 50)

    final_probs = np.zeros(len(test_df))
    for k, w in norm_weights.items():
        final_probs += w * family_predictions[k]

    print(f"\nApplying optimal operating decision threshold: {args.threshold:.2f}")
    final_preds = (final_probs >= args.threshold).astype(int)

    sub_df = pd.DataFrame({
        'image_id': test_df['image_id'],
        'label': final_preds
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    sub_df.to_csv(args.output_csv, index=False)
    print(f"\nSubmission saved to: {args.output_csv}")

    # Run strict validation
    validate_submission_format(sub_df, test_df)

if __name__ == "__main__":
    main()
