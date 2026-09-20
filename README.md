# Lunar Surface Classification: Depth vs. Rise

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A physics-aware deep learning framework for binary classification of lunar surface features into **Depth** (craters / depressions, class `0`) and **Rise** (mounds / boulders / elevated terrain, class `1`), achieving a cross-validated **Balanced Accuracy of 0.6720**.

---

## 🚀 Approach & Key Innovations

Distinguishing lunar craters from elevated mounds in single-monocular grayscale imagery is inherently ill-posed due to the **crater-dome crater illusion** caused by varying solar illumination angles. We address this using a physics-grounded pipeline:

1. **Physics-Based Illumination Normalization**:
   - Each lunar crop is rotated counter-clockwise by its exact negative solar azimuth angle ($-\text{sun\_azimuth\_angle}$) using edge-replication padding (`cv2.BORDER_REPLICATE`).
   - This aligns all incident shadows horizontally across the entire dataset, creating invariant shadow-casting physics (+13.40% boost in balanced accuracy).

2. **Topographic Shape-Relief Representations**:
   - Extracted normalized Sobel edge gradients as an illumination-invariant proxy for surface normal derivatives.
   - Evaluated both stacked 3-channel representations `[Intensity, Edge, Intensity]` and late-fusion `DualBranchResNet` architectures.

3. **Class Imbalance Mitigation**:
   - Lunar topography is inherently skewed (~63.7% Rise vs ~36.3% Depth).
   - Applied **`WeightedRandomSampler`** (50/50 balanced batches) coupled with **Focal Loss ($\gamma = 2.0$)** to penalize hard-to-classify Rise topography.

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
├── train.py                     # Consolidated 5-fold training pipeline
├── inference.py                 # Multi-model ensemble inference with TTA
├── requirements.txt             # Python dependencies
├── .gitignore                   # Excludes data, model weights, and logs
└── README.md                    # Project documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/lunar-surface-classification.git
   cd lunar-surface-classification
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏋️ Training

To reproduce the full 5-fold cross-validation training across all 7 model families:

```bash
python train.py --data_dir ./data --models_dir ./models --batch_size 48 --epochs_head 2 --epochs_ft 6 --model_family all
```

Options:
- `--data_dir`: Directory containing `train_metadata.csv` and `train_images/`
- `--models_dir`: Target directory for saved `.pt` model checkpoints
- `--model_family`: Model family to train (`all`, `convnext`, `resnet34`, `effnet_b2`, `dual_branch`, `resnet18`, `effnet_b0`)

---

## 🔮 Inference & Submission Generation

To run multi-model ensemble inference with Test-Time Augmentation on evaluation images:

```bash
python inference.py --data_dir ./data --models_dir ./models --output_csv ./submission.csv --threshold 0.52
```

### Download Pretrained Model Weights
All 33 trained checkpoint models are available for download here:
- **Google Drive Weights Archive**: [lunar_ensemble_weights.zip (1.77 GB)](https://drive.google.com/file/d/1Kr22X-Kl6xe7h3g-E8021LdU3dPL0aix/view?usp=sharing)
*(Download and extract the checkpoints into the `./models/` directory before running inference).*

---

## 📋 Submission Verification Checklist

Every generated `submission.csv` is strictly verified against competition specifications:
- Exactly 2,000 prediction rows (+ 1 header row = 2,001 lines)
- Column format: strictly `image_id,label`
- 0 missing, null, or NaN values
- Binary labels strictly in `{0, 1}`
- 100% ID alignment with `test_metadata.csv` in identical sequential order
