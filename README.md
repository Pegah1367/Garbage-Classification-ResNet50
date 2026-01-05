# ♻️ Smart Waste Sorting — Garbage Image Classification (ResNet-50 Transfer Learning)

An end-to-end computer vision project that classifies household waste images into **6 categories** — **cardboard, glass, metal, paper, plastic, trash** — using **transfer learning with ResNet-50**.  
The pipeline covers **data ingestion (Kaggle)**, **label harmonization**, **stratified splitting**, **augmentation**, **two-stage training (warm-up + fine-tune)**, **model evaluation**, and **interactive demos (Gradio + image upload)**.

---

## Why this project matters

Misclassification in recycling streams increases cost and contamination. This project demonstrates a practical ML workflow for **automated waste recognition**, showing how to take a public dataset and produce a **deployment-ready classifier** with measurable performance and a usable demo interface.

---

## Key Results

- **Dataset size:** **5,047 images** after filtering + label merge (`white-glass → glass`)
- **Split:** **4,287 train (85%) / 760 test (15%)** with per-class ratio preserved
- **Model:** ImageNet-pretrained **ResNet-50** + custom classification head
- **Performance:** Achieved **~0.96 test accuracy** on the held-out test set (760 images)

> Tip: Replace “~0.96” with your exact final score if you want the README to match your final run output exactly.

---

## Approach (What I built)

### 1) Data ingestion & label engineering
- Downloaded dataset using the **Kaggle API**
- Kept only target classes and **merged `white-glass` into `glass`**
- Built a clean dataframe of:
  - `imgPath` (image file path)
  - `label` (final class)

### 2) Stratified train/test split (custom)
Created a custom split function to ensure **each class keeps the same ratio** in train and test sets (avoids biased evaluation).

### 3) Data augmentation (train only)
To improve generalization, applied augmentation during training:
- rotation, flips, zoom
- rescaling to `[0, 1]`
- image size standardized to **224×224**

### 4) Transfer learning with ResNet-50 (two-stage training)
**Stage A — Warm-up**
- froze most of ResNet-50
- trained a new head until validation stabilized

**Stage B — Fine-tuning**
- unfroze deeper layers
- trained with a smaller learning rate to refine features for this dataset

### 5) Evaluation & analysis
- Confusion matrix visualization
- Per-class precision/recall/F1 report
- Random qualitative prediction grids (correct vs incorrect highlighted)

### 6) Demo options
- Batch prediction grids
- Upload-an-image classification
- **Gradio** app for interactive inference

---

## Model Architecture

**Backbone:** ResNet-50 (ImageNet, `include_top=False`)  
**Head:** GlobalAveragePooling → Dense(256, ReLU, L2) → BatchNorm → Dropout(0.3) → Softmax(6)

This avoids large `Flatten()` layers and improves generalization by using **Global Average Pooling**.

---

## Tech Stack

- **Python**, **TensorFlow / Keras**
- NumPy, Pandas, OpenCV
- Scikit-learn (metrics)
- Matplotlib / Seaborn (visuals)
- Gradio (optional demo)


