"""
Phase 1: Problem Definition & Dataset Collection
------------------------------------------------
- Defines the target medical problem: Automated multi-class Chest X-Ray disease detection.
- Target classes: Normal, Pneumonia, Tuberculosis.
- Verifies directory integrity, counts images per class, and triggers sample dataset creation if empty.
"""

import os
from pathlib import Path
from config import RAW_DATA_DIR, CLASS_NAMES, MEDICAL_DISCLAIMER
from dataset_generator import create_sample_dataset

def run_phase1():
    print("\n" + "="*70)
    print(" PHASE 1: PROBLEM DEFINITION & DATASET COLLECTION")
    print("="*70)
    print(MEDICAL_DISCLAIMER)
    print("\nProblem Statement:")
    print("   Detect lung diseases from chest X-Ray images using Deep Learning & ML.")
    print(f"   Target Classes: {', '.join(CLASS_NAMES)}")

    # Check dataset existence
    missing_classes = []
    class_counts = {}

    for cls in CLASS_NAMES:
        cls_dir = RAW_DATA_DIR / cls
        if not cls_dir.exists():
            missing_classes.append(cls)
        else:
            images = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg"))
            class_counts[cls] = len(images)
            if len(images) == 0:
                missing_classes.append(cls)

    if missing_classes or sum(class_counts.values()) == 0:
        print("\nRaw dataset missing or incomplete. Generating sample dataset...")
        create_sample_dataset(samples_per_class=40)
        
        # Re-count
        for cls in CLASS_NAMES:
            cls_dir = RAW_DATA_DIR / cls
            images = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg"))
            class_counts[cls] = len(images)

    print("\nDataset Collection Summary:")
    print(f"   Dataset Location: {RAW_DATA_DIR}")
    for cls, count in class_counts.items():
        print(f"   - Class '{cls}': {count} images")

    print(f"   Total Images Collected: {sum(class_counts.values())}")
    print("Phase 1 Complete!\n")
    return class_counts

if __name__ == "__main__":
    run_phase1()
