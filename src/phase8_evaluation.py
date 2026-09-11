"""
Phase 8: Model Evaluation
-------------------------
- Evaluates the trained Deep Learning ResNet model on the held-out Test dataset.
- Computes Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
- Generates Confusion Matrix plot -> outputs/figures/phase8_confusion_matrix.png.
- Generates Multi-class ROC-AUC curve -> outputs/figures/phase8_roc_auc_curve.png.
- Saves text evaluation report -> outputs/reports/phase8_evaluation_report.txt.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from config import SPLIT_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, CLASS_NAMES, NUM_CLASSES

def run_phase8():
    print("\n" + "="*70)
    print(" 📈 PHASE 8: COMPREHENSIVE MODEL EVALUATION")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ResNet model
    weights_path = MODELS_DIR / "resnet_model.pth"
    model = models.resnet18()
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, NUM_CLASSES))

    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("   Loaded ResNet model weights for evaluation.")
    else:
        print("⚠️ Model weights missing. Running evaluation with initialized network.")

    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = datasets.ImageFolder(SPLIT_DATA_DIR / "test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # 1. Overall Metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    print("\n📋 Comprehensive Performance Summary:")
    print(f"   Overall Accuracy : {acc * 100:.2f}%")
    print(f"   Weighted Precision: {prec:.4f}")
    print(f"   Weighted Recall   : {rec:.4f}")
    print(f"   Weighted F1-Score : {f1:.4f}")

    print("\n📊 Per-Class Classification Report:")
    clf_report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print(clf_report)

    # Save Report File
    report_file = REPORTS_DIR / "phase8_evaluation_report.txt"
    with open(report_file, "w") as f:
        f.write("Smart Healthcare - Medical Image Disease Detection Evaluation Report\n")
        f.write("="*70 + "\n")
        f.write(f"Accuracy : {acc*100:.2f}%\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")
        f.write(f"F1-Score : {f1:.4f}\n\n")
        f.write(clf_report)
    print(f"✅ Saved Text Report: {report_file}")

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title("Phase 8: Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Disease Class")
    ax.set_ylabel("True Disease Class")
    plt.tight_layout()
    cm_path = FIGURES_DIR / "phase8_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"✅ Saved Confusion Matrix Heatmap: {cm_path}")

    # 3. Multi-Class One-vs-Rest ROC-AUC Plot
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["teal", "crimson", "darkorange"]

    for i in range(NUM_CLASSES):
        if NUM_CLASSES == 2:
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], label=f"ROC Curve (AUC = {roc_auc:.3f})")
            break
        else:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=2, label=f"Class '{CLASS_NAMES[i]}' (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance Level (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Phase 8: Receiver Operating Characteristic (ROC-AUC) Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")

    plt.tight_layout()
    roc_path = FIGURES_DIR / "phase8_roc_auc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"✅ Saved ROC-AUC Curves Plot: {roc_path}")
    print("✅ Phase 8 Complete!\n")

    return {"accuracy": acc, "f1": f1, "confusion_matrix": cm}

if __name__ == "__main__":
    run_phase8()
