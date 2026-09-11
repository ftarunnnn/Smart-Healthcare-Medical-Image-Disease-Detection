"""
Phase 4: Data Augmentation
---------------------------
- Applies medically appropriate transformations: Rotation (±10°), Zoom (±5%), Horizontal Flip, Translation.
- Avoids vertical flipping (anatomical orientation must remain intact).
- Generates an Augmentation Preview Grid to illustrate training diversity.
- Saves report to outputs/figures/phase4_augmentation_grid.png.
"""

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from config import PROCESSED_DATA_DIR, CLASS_NAMES, FIGURES_DIR, RANDOM_SEED

def get_augmentation_pipeline():
    """
    Returns torchvision transforms for medical image data augmentation.
    """
    return T.Compose([
        T.RandomRotation(degrees=(-10, 10)),
        T.RandomResizedCrop(size=(224, 224), scale=(0.95, 1.05), ratio=(0.95, 1.05)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1)
    ])


def run_phase4():
    print("\n" + "="*70)
    print(" 🔄 PHASE 4: DATA AUGMENTATION & REGULARIZATION")
    print("="*70)
    
    transform = get_augmentation_pipeline()

    print("\n📋 Medical Data Augmentation Strategy:")
    print("   - Random Rotation: ±10 degrees (simulates slight patient positioning variance)")
    print("   - Random Zoom / Crop: 95% - 105% (simulates field-of-view differences)")
    print("   - Horizontal Flip: 50% chance (safe for symmetrical chest anatomy)")
    print("   - Brightness / Contrast Jitter: ±10% (simulates exposure settings)")
    print("   - ❌ Vertical Flip: EXCLUDED (violates anatomical top-down orientation)")

    # Find sample image from processed data
    sample_img_path = None
    for cls in CLASS_NAMES:
        cls_dir = PROCESSED_DATA_DIR / cls
        files = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg"))
        if files:
            sample_img_path = files[0]
            break

    if sample_img_path:
        img_orig = Image.open(sample_img_path).convert("RGB")
        
        fig, axes = plt.subplots(1, 6, figsize=(18, 3))
        axes[0].imshow(img_orig)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        for i in range(1, 6):
            aug_img = transform(img_orig)
            axes[i].imshow(aug_img)
            axes[i].set_title(f"Augmented #{i}")
            axes[i].axis("off")
            
        plt.suptitle("Phase 4: Medical Image Data Augmentation Variations", fontsize=14, fontweight="bold")
        out_fig = FIGURES_DIR / "phase4_augmentation_grid.png"
        plt.savefig(out_fig, dpi=300)
        plt.close()
        print(f"✅ Saved Augmentation Preview Grid: {out_fig}")

    print("✅ Phase 4 Complete!\n")

if __name__ == "__main__":
    run_phase4()
