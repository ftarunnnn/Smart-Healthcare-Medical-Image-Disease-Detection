"""
Phase 7: Feature Extraction + Classical ML
-------------------------------------------
- Uses trained ResNet18 as a Deep Feature Extractor (512-dim embeddings).
- Trains Classical Machine Learning Classifiers (SVM, Random Forest, XGBoost).
- Evaluates classifier performance on extracted features.
- Saves trained models to outputs/models/.
- Generates classifier comparison bar chart to outputs/figures/phase7_ml_classifier_comparison.png.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from config import SPLIT_DATA_DIR, MODELS_DIR, FIGURES_DIR, NUM_CLASSES, RANDOM_SEED

def extract_features(model, dataloader, device):
    """
    Extracts 512-dimensional bottleneck feature vectors from the ResNet backbone.
    """
    model.eval()
    # Create feature extractor hook up to avgpool
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()

    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            feats = feature_extractor(images)  # Shape: (B, 512, 1, 1)
            feats = torch.flatten(feats, 1)    # Shape: (B, 512)
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())

    X = np.vstack(features_list)
    y = np.concatenate(labels_list)
    return X, y


def run_phase7():
    print("\n" + "="*70)
    print(" 🔬 PHASE 7: FEATURE EXTRACTION + CLASSICAL ML CLASSIFIERS")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load ResNet model
    weights_path = MODELS_DIR / "resnet_model.pth"
    model = models.resnet18()
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, NUM_CLASSES))

    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("   Successfully loaded trained ResNet weights for Feature Extraction.")
    else:
        print("⚠️ Saved ResNet weights not found. Using pretrained ImageNet weights for extraction.")

    model = model.to(device)

    # Data Loaders
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(SPLIT_DATA_DIR / "train", transform=transform)
    test_ds = datasets.ImageFolder(SPLIT_DATA_DIR / "test", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    print("\n⚡ Extracting Deep Bottleneck Representations (512-dim vectors)...")
    X_train, y_train = extract_features(model, train_loader, device)
    X_test, y_test = extract_features(model, test_loader, device)

    print(f"   Extracted Feature Matrix (Train): {X_train.shape}")
    print(f"   Extracted Feature Matrix (Test) : {X_test.shape}")

    # Define Classifiers
    classifiers = {
        "SVM (RBF Kernel)": SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_SEED, eval_metric="mlogloss")
    }

    results = {}
    print("\n🎯 Training & Evaluating Classical ML Classifiers:")
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        results[name] = {"accuracy": acc, "f1": f1}
        print(f"   - {name:<20} | Test Accuracy: {acc*100:.2f}% | F1-Score: {f1:.4f}")

        # Save model
        clean_name = name.lower().split()[0]
        joblib.dump(clf, MODELS_DIR / f"{clean_name}_classifier.joblib")

    # Save Comparison Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    accs = [results[n]["accuracy"] * 100 for n in names]
    f1s = [results[n]["f1"] * 100 for n in names]

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width/2, accs, width, label="Accuracy (%)", color="navy")
    ax.bar(x + width/2, f1s, width, label="F1-Score (%)", color="crimson")

    ax.set_ylabel("Score (%)")
    ax.set_title("Phase 7: Classical ML Classifiers on Deep Feature Embeddings")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim([0, 105])
    ax.legend()

    for i in range(len(names)):
        ax.text(i - width/2, accs[i] + 1, f"{accs[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
        ax.text(i + width/2, f1s[i] + 1, f"{f1s[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out_fig = FIGURES_DIR / "phase7_ml_classifier_comparison.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()

    print(f"\n✅ Saved ML Comparison Chart: {out_fig}")
    print("✅ Phase 7 Complete!\n")
    return results

if __name__ == "__main__":
    run_phase7()
