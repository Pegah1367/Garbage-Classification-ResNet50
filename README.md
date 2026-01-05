
# ♻️ Garbage Image Classification (ResNet-50 Transfer Learning)

A computer vision project that classifies waste images into **6 categories** (**paper, plastic, cardboard, glass, metal, trash**) using **transfer learning with ResNet-50**. The workflow covers dataset preparation, stratified splitting, augmentation, warm-up training, fine-tuning, evaluation (classification report + confusion matrix), and interactive demo options.

---

## Project Highlights

- **Problem:** Multi-class image classification for recyclable waste sorting (6 classes). :contentReference[oaicite:0]{index=0}  
- **Dataset:** Enhanced Kaggle dataset mapped to the same 6 target classes, totaling **5,047 images**. :contentReference[oaicite:1]{index=1}  
- **Split:** Custom **stratified** split — **85% train / 15% test**  
  - Train: **4,287** images  
  - Test: **760** images :contentReference[oaicite:2]{index=2}  
- **Model:** ResNet-50 (ImageNet pretrained) + custom head (GAP → Dense(256, L2) → BN → Dropout(0.3) → Softmax). :contentReference[oaicite:3]{index=3}  
- **Training Strategy:** Two-stage transfer learning  
  1) **Warm-up:** freeze backbone except last ~10 layers :contentReference[oaicite:4]{index=4}  
  2) **Fine-tune:** unfreeze last ~50 layers with small LR (1e-5) :contentReference[oaicite:5]{index=5}  

---

## Results (Test Set)

- The model achieved **strong and consistent performance across all six classes**, with most class metrics in the **0.93–0.99** range and overall accuracy around **0.95** (report-based summary). :contentReference[oaicite:6]{index=6}  
- Confusions mainly occurred between visually similar categories (e.g., **glass vs metal**, some **plastic** → metal/glass). :contentReference[oaicite:7]{index=7}  

> Note: If you want, I can format your exact numeric results (accuracy, loss, full classification report) into a clean “Results” block once you paste your final run outputs.

---

## Dataset & Preprocessing

### Classes
- cardboard
- glass
- metal
- paper
- plastic
- trash :contentReference[oaicite:8]{index=8}  

### Augmentation (Train Only)
To improve robustness and reduce overfitting, augmentation is applied only to the **training** set (test remains “real / untouched” for fair evaluation). :contentReference[oaicite:9]{index=9}  

- Rescale: 1/255  
- Rotation: 45°  
- Horizontal + vertical flip  
- Zoom: (1.0, 1.2)  
- Target size: 224×224 :contentReference[oaicite:10]{index=10}  

---

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV, NumPy, Pandas, Matplotlib, Seaborn
- Kaggle API (dataset download)
- Optional UI demo: Gradio




