"""
Consolidated Training Pipeline for Lunar Surface Classification.
Implements:
1. Physics-based sun-azimuth rotation & topographic relief feature extraction.
2. Class-balanced sampling (WeightedRandomSampler) & Focal Loss (gamma=2.0).
3. Two-stage fine-tuning with Cosine Annealing learning rate schedule.
4. Stratified 5-Fold Cross-Validation across all 7 model families.
5. Out-of-fold threshold optimization maximizing Balanced Accuracy.
"""

import os
import argparse
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.dataset import LunarSurfaceDataset
from src.losses import FocalLoss
from src.metrics import compute_metrics, find_optimal_threshold
from src.models import (
    DualBranchResNet,
    build_resnet18,
    build_resnet34,
    build_efficientnet_b0,
    build_efficientnet_b2,
    build_convnext_tiny
)

def get_balanced_loader(dataset: LunarSurfaceDataset, batch_size: int = 48) -> DataLoader:
    labels = dataset.df['label'].values
    count_0 = (labels == 0).sum()
    count_1 = (labels == 1).sum()
    weights = [1.0 / count_0 if y == 0 else 1.0 / count_1 for y in labels]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)

def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_val: np.ndarray,
    device: torch.device,
    is_dual: bool = False,
    epochs_head: int = 2,
    epochs_ft: int = 6,
    save_path: str = None
) -> tuple:
    criterion = FocalLoss(gamma=2.0)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    # Stage 1: Train classification head
    if hasattr(model, 'fusion_head'):
        for p in model.parameters(): p.requires_grad = False
        for p in model.fusion_head.parameters(): p.requires_grad = True
        head_params = model.fusion_head.parameters()
    elif hasattr(model, 'fc'):
        for p in model.parameters(): p.requires_grad = False
        for p in model.fc.parameters(): p.requires_grad = True
        head_params = model.fc.parameters()
    elif hasattr(model, 'classifier'):
        for p in model.parameters(): p.requires_grad = False
        for p in model.classifier.parameters(): p.requires_grad = True
        head_params = model.classifier.parameters()
    else:
        head_params = model.parameters()
        
    optimizer_head = torch.optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, epochs_head + 1):
        model.train()
        for batch in train_loader:
            optimizer_head.zero_grad()
            if is_dual:
                (x_opt, x_rel), targets, _ = batch
                x_opt, x_rel = x_opt.to(device, non_blocking=True), x_rel.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    outputs = model(x_opt, x_rel)
                    loss = criterion(outputs, targets)
            else:
                inputs, targets, _ = batch
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer_head)
            scaler.update()

    # Stage 2: Fine-tune all layers with Cosine Annealing
    for p in model.parameters(): p.requires_grad = True
    optimizer_ft = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=epochs_ft, eta_min=1e-6)
    
    best_bal_acc = -1.0
    best_state = None
    best_val_probs = None
    
    for epoch in range(1, epochs_ft + 1):
        model.train()
        for batch in train_loader:
            optimizer_ft.zero_grad()
            if is_dual:
                (x_opt, x_rel), targets, _ = batch
                x_opt, x_rel = x_opt.to(device, non_blocking=True), x_rel.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    outputs = model(x_opt, x_rel)
                    loss = criterion(outputs, targets)
            else:
                inputs, targets, _ = batch
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer_ft)
            scaler.update()
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_probs = []
        with torch.no_grad():
            for batch in val_loader:
                if is_dual:
                    (x_opt, x_rel), _, _ = batch
                    x_opt, x_rel = x_opt.to(device, non_blocking=True), x_rel.to(device, non_blocking=True)
                    with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                        out = model(x_opt, x_rel)
                else:
                    inputs, _, _ = batch
                    inputs = inputs.to(device, non_blocking=True)
                    with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                        out = model(inputs)
                val_probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
                
        val_probs = np.array(val_probs)
        thresh, bal_acc = find_optimal_threshold(y_val, val_probs)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_state = copy.deepcopy(model.state_dict())
            best_val_probs = val_probs

    if save_path and best_state is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_state, save_path)
        
    best_th, _ = find_optimal_threshold(y_val, best_val_probs)
    preds = (best_val_probs >= best_th).astype(int)
    metrics = compute_metrics(y_val, preds)
    metrics['best_threshold'] = best_th
    return metrics, best_val_probs

def main():
    parser = argparse.ArgumentParser(description="Lunar Surface Classification Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing metadata CSVs and images")
    parser.add_argument("--models_dir", type=str, default="./models", help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size for training")
    parser.add_argument("--epochs_head", type=int, default=2, help="Epochs for training classifier head")
    parser.add_argument("--epochs_ft", type=int, default=6, help="Epochs for fine-tuning entire network")
    parser.add_argument("--model_family", type=str, default="all", choices=["all", "convnext", "resnet34", "effnet_b2", "dual_branch", "resnet18", "effnet_b0"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    meta_path = os.path.join(args.data_dir, "train_metadata.csv")
    train_images_dir = os.path.join(args.data_dir, "train_images")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}. Place train_metadata.csv in {args.data_dir}")

    df = pd.read_csv(meta_path)
    
    # Generate or load 5 stratified folds
    folds_path = os.path.join(args.data_dir, "train_folds.csv")
    if os.path.exists(folds_path):
        df = pd.read_csv(folds_path)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        df['fold'] = -1
        for f, (_, val_idx) in enumerate(skf.split(df, df['label'])):
            df.loc[val_idx, 'fold'] = f
        df.to_csv(folds_path, index=False)
        print(f"Created stratified 5-fold split at {folds_path}")

    model_registry = {
        'resnet18_optical': (lambda: build_resnet18(pretrained=True), 'optical', False),
        'efficientnet_b0_optical': (lambda: build_efficientnet_b0(pretrained=True), 'optical', False),
        'dual_branch_resnet18': (lambda: DualBranchResNet('resnet18', pretrained=True), 'dual', True),
        'resnet18_stacked': (lambda: build_resnet18(pretrained=True), 'stacked', False),
        'resnet34_stacked': (lambda: build_resnet34(pretrained=True), 'stacked', False),
        'effnet_b2_stacked': (lambda: build_efficientnet_b2(pretrained=True), 'stacked', False),
        'convnext_tiny_stacked': (lambda: build_convnext_tiny(pretrained=True), 'stacked', False)
    }

    if args.model_family != "all":
        model_registry = {k: v for k, v in model_registry.items() if args.model_family in k}

    results = []
    for model_name, (builder_fn, mode, is_dual) in model_registry.items():
        print(f"\n{'=' * 60}\nTraining Model Family: {model_name}\n{'=' * 60}")
        fold_metrics = []
        for fold in range(5):
            print(f"\n--- Fold {fold} ---")
            train_df = df[df['fold'] != fold].reset_index(drop=True)
            val_df = df[df['fold'] == fold].reset_index(drop=True)
            
            train_ds = LunarSurfaceDataset(train_df, train_images_dir, is_train=True, mode=mode)
            val_ds = LunarSurfaceDataset(val_df, train_images_dir, is_train=False, mode=mode)
            
            train_loader = get_balanced_loader(train_ds, batch_size=args.batch_size)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            
            model = builder_fn().to(device)
            save_path = os.path.join(args.models_dir, f"{model_name}_fold_{fold}.pt")
            
            metrics, _ = train_one_fold(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                y_val=val_df['label'].values,
                device=device,
                is_dual=is_dual,
                epochs_head=args.epochs_head,
                epochs_ft=args.epochs_ft,
                save_path=save_path
            )
            print(f"Fold {fold} Finished -> Bal Acc: {metrics['balanced_accuracy']:.4f} | Opt Thresh: {metrics['best_threshold']:.2f}")
            fold_metrics.append(metrics)
            
        bal_accs = [m['balanced_accuracy'] for m in fold_metrics]
        mean_bal = np.mean(bal_accs)
        std_bal = np.std(bal_accs)
        print(f"\n>> {model_name} 5-Fold Balanced Accuracy: {mean_bal:.4f} +/- {std_bal:.4f} <<")
        results.append({
            'model': model_name,
            'balanced_accuracy_mean': mean_bal,
            'balanced_accuracy_std': std_bal
        })

    print("\nTraining complete! Checkpoints saved to:", args.models_dir)

if __name__ == "__main__":
    main()
