# Lunar Surface Classification: Depth vs. Rise

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A physics-aware deep learning framework for binary classification of lunar surface features into **Depth** (craters / depressions, class `0`) and **Rise** (mounds / boulders / elevated terrain, class `1`), achieving a cross-validated **Balanced Accuracy of 0.6720**.

---

## 💻 Hardware Requirements

| Component | Minimum Specification | Recommended Specification (Tested) |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GPU with CUDA support | **NVIDIA GeForce RTX 3050 (6GB VRAM)** or higher |
| **VRAM** | 4 GB VRAM | **6 GB+ VRAM** |
| **System RAM**| 8 GB RAM | **16 GB RAM** |
| **Storage** | ~5 GB free disk space | **10 GB+ SSD** (1.8 GB weights + 1.2 GB dataset) |
| **OS** | Windows 10/11 or Ubuntu 20.04+ | **Windows 11 / Linux (x86_64)** |
| **CUDA / Driver** | CUDA 11.8+ / 12.x | **CUDA 12.6 / 12.9** |

---

## 🚀 Approach & Key Innovations

Distinguishing lunar craters from elevated mounds in single-monocular grayscale imagery is inherently ill-posed due to the **crater-dome optical illusion** caused by varying solar illumination angles. We address this using a physics-grounded pipeline:

1. **Physics-Based Illumination Normalization**:
   - Each lunar crop is rotated counter-clockwise by its exact negative solar azimuth angle ($-\text{sun\_azimuth\_angle}$) using edge-replication padding (`cv2.BORDER_REPLICATE`).
   - This aligns all incident shadows horizontally across the entire dataset, creating invariant shadow-casting physics (+13.40% boost in balanced accuracy).

2. **Topographic Shape-Relief Representations**:
   - Extracted normalized Sobel edge gradients as an illumination-invariant proxy for surface normal derivatives.
   - Evaluated both stacked 3-channel representations `[Intensity, Edge, Intensity]` and late-fusion `DualBranchResNet` architectures.

3. **Class Imbalance Mitigation**:
   - Lunar topography is naturally skewed (~63.7% Rise vs ~36.3% Depth).
   - Applied **`WeightedRandomSampler`** (50/50 balanced batches) coupled with **Focal Loss ($\gamma = 2.0$)** to heavily penalize hard-to-classify Rise topography.

4. **Multi-Model Architectural Diversity**:
   - Trained 7 diverse model families across Stratified 5-Fold Cross-Validation:
     - **ConvNeXt-Tiny** (Stacked Relief): $7 \times 7$ depthwise convolutions for large receptive field
     - **EfficientNet-B0 & EfficientNet-B2** (Compound scaling)
     - **ResNet-18 & ResNet-34** (Residual learning)
     - **Dual-Branch ResNet-18** (Isolated optical vs. relief feature branches)

5. **PR-Curve Threshold Optimization & TTA Ensembling**:
   - Operating thresholds were swept from 0.10 to 0.90 to strictly maximize **Balanced Accuracy** (optimal operating point = `0.52`).
   - Multi-model probability fusion with horizontal-flip Test-Time Augmentation (TTA).

---

## 📊 Cross-Validation Performance Comparison

| Model Architecture / Backbone | Balanced Accuracy (Mean ± Std) | Accuracy | Recall-0 (Depth) | Recall-1 (Rise) | F1-Score | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ensemble (Full 7-Family Fusion + TTA)** | **0.6720** | **0.6315** | **0.8190** | **0.5250** | **0.6435** | **0.52** |
| ConvNeXt-Tiny (Stacked Relief + Focal Loss) | 0.6593 ± 0.0139 | 0.6177 | 0.8120 | 0.5067 | 0.6275 | 0.50 |
| EfficientNet-B0 (5-Fold CV) | 0.6583 ± 0.0083 | 0.6183 | 0.8048 | 0.5118 | 0.6300 | 0.56 |
| ResNet-18 (5-Fold CV) | 0.6501 ± 0.0135 | 0.6129 | 0.7859 | 0.5142 | 0.6249 | 0.54 |
| ResNet-18 (Stacked Relief + Focal Loss) | 0.6444 ± 0.0094 | 0.6127 | 0.7604 | 0.5284 | 0.6322 | 0.51 |
| EfficientNet-B2 (Stacked Relief + Focal Loss) | 0.6441 ± 0.0155 | 0.5949 | 0.8241 | 0.4640 | 0.5925 | 0.54 |
| Dual-Branch ResNet-18 (5-Fold CV) | 0.6402 ± 0.0114 | 0.6025 | 0.7813 | 0.4990 | 0.6115 | 0.52 |
| ResNet-34 (Stacked Relief + Focal Loss) | 0.6381 ± 0.0112 | 0.6020 | 0.7702 | 0.5060 | 0.6179 | 0.52 |
| *ResNet-18 (No Physics Baseline)* | *0.5161* | *0.5691* | *0.3222* | *0.7100* | *0.6772* | *0.57* |

---

## 📦 Project Structure

```
lunar-surface-classification/
├── src/
│   ├── __init__.py              # Package initializer
│   ├── preprocessing.py         # Physics rotation & Sobel relief extraction
│   ├── models.py                # DualBranch, ResNet, EffNet, ConvNeXt builders
│   ├── dataset.py               # LunarSurfaceDataset with augmentation
│   ├── losses.py                # Focal Loss (gamma=2.0)
│   └── metrics.py               # Balanced Accuracy & PR-curve threshold sweeps
├── prepare_data.py              # Automated download, unzipping, verification & folds creation
├── train.py                     # Consolidated 5-fold CV training pipeline
├── inference.py                 # Multi-model ensemble inference with TTA
├── requirements.txt             # Exact pinned Python dependencies
├── .gitignore                   # Excludes data, model weights, and logs
└── README.md                    # Project documentation & execution guide
```

---

## 🛠️ Step-by-Step Installation & Setup

### Step 1: Clone Repository & Set Up Virtual Environment

```bash
git clone https://github.com/khushichandak27/lunar-surface-classification.git
cd lunar-surface-classification

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install exact pinned dependencies
pip install -r requirements.txt
```

---

### Step 2: Data Preparation & Verification

Run the automated data preparation script:

```bash
python prepare_data.py --data_dir ./data
```

This script:
1. Automatically downloads the competition image archives from Google Drive if not already downloaded.
2. Extracts and organizes crops into `./data/train_images` (7,854 crops) and `./data/eval_images` (2,000 crops).
3. Validates file integrity, dimensions ($256 \times 256$), and confirms solar azimuth angles are in $[0^\circ, 360^\circ]$.
4. Creates leakage-free Stratified 5-Fold split definitions (`./data/train_folds.csv`).

*(If you already have `train_metadata.csv`, `test_metadata.csv`, and the extracted image folders, place them inside `./data/` and `prepare_data.py` will verify them instantly).*

---

### Step 3: Download Pretrained Model Weights

Download the pre-packaged archive containing all 33 winning model checkpoints:
- **Google Drive Weights Archive (1.77 GB)**: [lunar_ensemble_weights.zip](https://drive.google.com/file/d/1Kr22X-Kl6xe7h3g-E8021LdU3dPL0aix/view?usp=sharing)

Extract the `.pt` files directly into the `./models/` directory:
```
lunar-surface-classification/
└── models/
    ├── convnext_tiny_stacked_relief_fold_0.pt ... fold_2.pt
    ├── effnet_b2_stacked_relief_fold_0.pt ... fold_4.pt
    ├── resnet34_stacked_relief_fold_0.pt ... fold_4.pt
    ├── dual_branch_resnet18_fold_0.pt ... fold_4.pt
    ├── resnet18_stacked_relief_fold_0.pt ... fold_4.pt
    ├── efficientnet_b0_fold_0.pt ... fold_4.pt
    └── resnet18_fold_0.pt ... fold_4.pt
```

---

## 🔮 Standalone Single-Command Inference

To reproduce the winning competition `submission.csv` with a **single command**:

```bash
python inference.py
```

`inference.py` runs standalone out-of-the-box:
- Loads the 2,000 evaluation images and metadata from `./data`
- Automatically loads all 33 model checkpoints from `./models`
- Executes Test-Time Augmentation (TTA: original + horizontal flip)
- Fuses multi-model predictions with optimal architecture weights (ConvNeXt: 25%, EffNet-B2: 15%, ResNet-34: 15%, Dual-Branch: 15%, ResNet-18 Stacked: 10%, EffNet-B0: 10%, ResNet-18: 10%)
- Applies the validation-calibrated threshold (`0.52`)
- Saves `./submission.csv` and runs strict format validation checks

#### Optional Custom Inference Arguments:
```bash
python inference.py --data_dir /path/to/data --models_dir /path/to/models --output_csv ./submission.csv --threshold 0.52 --batch_size 32
```

---

## 🏋️ Model Retraining

To retrain any or all model families across 5-fold cross-validation:

```bash
# Train all 7 model families across 5 folds:
python train.py --data_dir ./data --models_dir ./models --batch_size 48 --epochs_head 2 --epochs_ft 6 --model_family all

# Or train a specific architecture family:
python train.py --model_family convnext
```

---

## 📋 Submission Verification Checklist

Every generated `submission.csv` automatically passes these strict checks:
- [x] Exactly 2,000 prediction rows (+ 1 header row = 2,001 lines)
- [x] Column header: strictly `image_id,label`
- [x] 0 missing, null, or NaN values
- [x] Binary labels strictly in `{0, 1}`
- [x] 100% ID alignment with `test_metadata.csv` in identical sequential order
