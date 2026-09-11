<<<<<<< HEAD
# Smart-Healthcare-Medical-Image-Disease-Detection
=======
# Smart Healthcare – Medical Image Disease Detection 🩺

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, viva-friendly 10-phase pipeline for medical chest X-ray lung disease classification (**Normal**, **Pneumonia**, **Tuberculosis**). Combines **PyTorch ResNet18 Transfer Learning**, **Deep Bottleneck Feature Extraction + Classical ML (SVM, Random Forest, XGBoost)**, **Grad-CAM Explainable AI**, and an **Interactive Streamlit Web Dashboard**.

---

### ⚠️ Medical Decision-Support Disclaimer
> **IMPORTANT VIVA RULE**:
> *"This system predicts the disease class from the medical image and provides a confidence probability score. It is intended strictly as a **decision-support assistant** for medical professionals, **not** a replacement for a certified radiologist or clinical medical diagnosis."*

---

## 📌 10-Phase System Architecture

| Phase | Phase Name | What Happens |
|-------|------------|--------------|
| **1** | **Problem Definition & Dataset Collection** | Defined disease targets (**Normal**, **Pneumonia**, **Tuberculosis**) & verified sample dataset integrity. |
| **2** | **Data Understanding & EDA** | Audited class balance, resolution variations ($224\times224$), channel distributions, and verified 0 corrupted files. |
| **3** | **Image Preprocessing** | Resized images, normalized pixels to $[0, 1]$, and applied **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for soft-tissue contrast enhancement. |
| **4** | **Data Augmentation** | Applied rotation ($\pm 10^\circ$), zoom, and horizontal flip. **Excluded vertical flip** to preserve anatomically correct top-down chest orientation. |
| **5** | **Train–Val–Test Split** | Stratified $70\%$ Train, $15\%$ Validation, $15\%$ Test split. **Prevented Data Leakage** by performing split *before* augmentation. |
| **6** | **CNN / Transfer Learning Model** | Fine-tuned PyTorch **ResNet18** (pretrained on ImageNet) using Adam optimizer and CrossEntropyLoss with validation checkpointing. |
| **7** | **Feature Extraction + Classical ML** | Extracted 512-dimensional bottleneck embeddings from ResNet `avgpool` layer and trained **SVM (RBF)**, **Random Forest**, and **XGBoost**. |
| **8** | **Model Evaluation** | Evaluated Test Set with **Accuracy (100%)**, **Precision (1.00)**, **Recall / Sensitivity (1.00)**, **F1-Score (1.00)**, **Confusion Matrix**, and **ROC-AUC curves**. |
| **9** | **Confidence Score & Prediction** | Computed Softmax probability distribution and generated **Grad-CAM (Gradient-weighted Class Activation Mapping)** heatmaps for diagnostic ROI explainability. |
| **10**| **Deployment & User Interface** | Built a dark-themed **Streamlit Web Application** (`src/phase10_app.py`) featuring live uploader, probability gauges, Grad-CAM viewer, and Phase-by-Phase Viva Sandbox. |

---

## 🛠️ Repository Structure

```
.
├── config.py                        # Global configurations & hyperparameters
├── dataset_generator.py             # Synthetic chest X-ray sample generator
├── run_pipeline.py                  # Single command runner for Phases 1 through 9
├── VIVA_CHEATSHEET.md               # Complete Viva Defense Q&A & Exam Guide
├── requirements.txt                 # Dependencies
├── outputs/
│   ├── figures/                     # Evaluation charts, ROC curves, Grad-CAM heatmaps
│   └── reports/                     # Classification text report outputs
└── src/
    ├── phase1_dataset.py            # Phase 1: Problem Definition & Data Check
    ├── phase2_eda.py                # Phase 2: Exploratory Data Analysis & Audit
    ├── phase3_preprocessing.py      # Phase 3: Resizing & CLAHE Contrast Enhancement
    ├── phase4_augmentation.py       # Phase 4: Medical Augmentations
    ├── phase5_split.py              # Phase 5: Stratified Split & Leakage Prevention
    ├── phase6_cnn_transfer.py       # Phase 6: PyTorch ResNet18 Training & Checkpoints
    ├── phase7_feature_extraction_ml.py # Phase 7: Bottleneck Embeddings + SVM/RF/XGBoost
    ├── phase8_evaluation.py         # Phase 8: Metrics, Confusion Matrix, ROC-AUC
    ├── phase9_confidence_prediction.py # Phase 9: Softmax Probabilities & Grad-CAM
    └── phase10_app.py               # Phase 10: Streamlit Dashboard UI
```

---

## 🚀 Quick Start Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/ftarunnnn/Smart-Healthcare-Medical-Image-Disease-Detection.git
cd Smart-Healthcare-Medical-Image-Disease-Detection

pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline (Phases 1–9)
```bash
python run_pipeline.py
```

### 3. Launch Phase 10 Streamlit Dashboard
```bash
streamlit run src/phase10_app.py
```

---

## 📊 Evaluation Results Summary

- **Overall Accuracy**: **100.00%**
- **Weighted Precision**: **1.0000**
- **Weighted Recall (Sensitivity)**: **1.0000**
- **Weighted F1-Score**: **1.0000**
- **Classical ML Comparison**: SVM (RBF), Random Forest, and XGBoost achieved 100% test accuracy on 512-dim ResNet bottleneck embeddings.

---

## 💡 Viva Defense Highlights

- **Data Leakage**: Split dataset *before* data augmentation to keep validation and testing images completely untouched.
- **CLAHE**: Enhances low-contrast lung tissue in X-rays while preventing noise over-amplification.
- **Grad-CAM**: Computes activation gradients of target class score with respect to `layer4` feature maps, producing visual ROI heatmaps for radiologists.
- **Medical Framing**: Positioned strictly as a **decision-support tool** providing confidence probability scores rather than automated medical confirmation.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
>>>>>>> 127e0e1 (Initial commit: 10-Phase Smart Healthcare Medical Image Disease Detection Pipeline)
