# image-based-waste-classification-with-transfer-learning

A CNN-based image classifier for household waste sorting support, 
built as a capstone project for the Applied AI/ML program at Tomorrow University.

The model predicts the dominant waste category in a single image across 
10 classes. The intended use is **recycling education and sorting assistance**, 
not autonomous waste-management decision-making.

---

## Project Structure

├── notebooks/
│ ├── 01_Waste_Class_EDA.ipynb # Dataset exploration, duplicate analysis, embeddings
│ └── 02_Waste_Class_Modeling.ipynb # Modeling pipeline, evaluation, Grad-CAM
├── artifacts/ # Saved models, metrics, plots
└── README.md


---

## Task

Single-label image classification across 10 waste categories:
`battery` · `biological` · `cardboard` · `clothes` · `glass` · 
`metal` · `paper` · `plastic` · `shoes` · `trash`

---
## Dataset

[Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/data) 
— hosted on Kaggle. Download and place in `data/` before running the notebooks.

> The dataset is not included in this repository due to file size.
---

## Modeling Approach

The modeling process was staged to evaluate whether transfer learning 
improves classification over a from-scratch CNN baseline.

| Model | Split | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| Baseline CNN | Validation | 0.633 | 0.619 | 0.631 |
| EfficientNetB0 frozen · Dense256 | Validation | 0.941 | 0.934 | 0.941 |
| EfficientNetB0 frozen · Dense512 | Validation | 0.943 | 0.934 | 0.944 |
| EfficientNetB0 fine-tuned · Dense256 | Validation | 0.944 | 0.935 | 0.944 |
| **Final: EfficientNetB0 frozen · Dense256** | **Test** | **0.946** | **0.942** | **0.946** |

The final model is the **frozen EfficientNetB0 with a Dense256 head** — 
selected for its balance of performance, macro F1 stability, and lower 
complexity compared to larger or fine-tuned alternatives.

---

## Key Findings

- The jump from baseline CNN → frozen EfficientNetB0 is the largest 
  single improvement; subsequent tuning yielded only marginal gains.
- Strongest classes: `clothes`, `biological`, `shoes`, `battery` (F1 ≥ 0.98) — 
  visually coherent categories with stable shapes or textures.
- Hardest classes: `trash`, `plastic`, `paper` — visually ambiguous, 
  overlapping, or heterogeneous categories.
- `trash` is a catch-all class without a stable visual prototype; 
  lower performance on it reflects a class-definition limitation, 
  not only model weakness.
- Grad-CAM confirms the model generally attends to object-relevant 
  regions rather than background shortcuts.

---

## Responsible Use

This model is intended as **decision support**, not an authoritative classifier.

- Predictions should be treated as category suggestions
- Low-confidence outputs should be flagged as uncertain
- High-risk categories (e.g. `battery`) warrant stricter thresholds
- Correct disposal always depends on local recycling rules

---

## Stack

- Python · TensorFlow / Keras · Scikit-learn
- EfficientNetB0 (ImageNet pretrained)
- Grad-CAM for interpretability
- Plotly · Matplotlib for visualization

---

## References

- Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition.
- Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.
- Srivastava, N. et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
- Russakovsky, O. et al. (2015). ImageNet Large Scale Visual Recognition Challenge.
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
- Selvaraju, R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
- Liang, J., Liu, Y., & Vlassov, V. (2023). The Impact of Background Removal on Performance of Neural Networks for Fashion Image Classification and Segmentation.

---

## How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/data) 
   and place it in `data/`

4. Run the notebooks in order:
   - `01_Waste_Class_EDA.ipynb` — dataset exploration and embedding analysis
   - `02_Waste_Class_Modeling.ipynb` — modeling pipeline, evaluation, and Grad-CAM
