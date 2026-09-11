"""
Phase 5: Train-Validation-Test Split
------------------------------------
- Performs stratified split: 70% Train, 15% Validation, 15% Test.
- Copies preprocessed images into dataset/splits/{train, val, test}/{class_name}.
- Prevents Data Leakage by performing split BEFORE data augmentation.
- Generates Split Summary chart to outputs/figures/phase5_split_distribution.png.
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from config import PROCESSED_DATA_DIR, SPLIT_DATA_DIR, CLASS_NAMES, RANDOM_SEED, FIGURES_DIR

def run_phase5():
    print("\n" + "="*70)
    print(" ✂️ PHASE 5: TRAIN - VALIDATION - TEST DATASET SPLIT")
    print("="*70)

    filepaths = []
    labels = []

    for cls in CLASS_NAMES:
        cls_dir = PROCESSED_DATA_DIR / cls
        files = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg"))
        for f in files:
            filepaths.append(f)
            labels.append(cls)

    filepaths = np.array(filepaths)
    labels = np.array(labels)

    # 1. Stratified Split: Train (70%), Temp (30%)
    train_files, temp_files, train_lbls, temp_lbls = train_test_split(
        filepaths, labels, test_size=0.30, random_state=RANDOM_SEED, stratify=labels
    )

    # 2. Stratified Split Temp: Val (15%), Test (15%)
    val_files, test_files, val_lbls, test_lbls = train_test_split(
        temp_files, temp_lbls, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_lbls
    )

    # Clean existing splits
    if SPLIT_DATA_DIR.exists():
        shutil.rmtree(SPLIT_DATA_DIR)

    splits = {
        "train": (train_files, train_lbls),
        "val": (val_files, val_lbls),
        "test": (test_files, test_lbls)
    }

    counts_summary = {split: {cls: 0 for cls in CLASS_NAMES} for split in splits}

    for split_name, (files, lbls) in splits.items():
        for f, l in zip(files, lbls):
            dest_dir = SPLIT_DATA_DIR / split_name / l
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dest_dir / f.name)
            counts_summary[split_name][l] += 1

    print("\n🛡️ Data Leakage Prevention Check:")
    print("   - Split executed at image index level BEFORE data augmentation.")
    print("   - Stratified distribution maintains class proportions across splits.")
    
    print("\n📊 Split Breakdown:")
    for split_name, counts in counts_summary.items():
        total = sum(counts.values())
        print(f"   - {split_name.upper()} Set ({total} images): {counts}")

    # Plot Split Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    for i, (split_name, counts) in enumerate(counts_summary.items()):
        vals = [counts[c] for c in CLASS_NAMES]
        ax.bar(x + i*width, vals, width, label=split_name.capitalize())

    ax.set_xlabel("Disease Class")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Phase 5: Stratified Dataset Split (Train 70% / Val 15% / Test 15%)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend()

    plt.tight_layout()
    out_fig = FIGURES_DIR / "phase5_split_distribution.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()

    print(f"✅ Saved Split Distribution Plot: {out_fig}")
    print("✅ Phase 5 Complete!\n")
    return counts_summary

if __name__ == "__main__":
    run_phase5()
