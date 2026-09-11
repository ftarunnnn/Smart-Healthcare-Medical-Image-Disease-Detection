"""
Phase 2: Data Understanding & EDA
----------------------------------
- Audits image classes, resolutions, aspect ratios, corrupted files.
- Generates Class Distribution bar chart and Sample Image Grids.
- Saves visual EDA report to outputs/figures/phase2_eda_summary.png.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
from config import RAW_DATA_DIR, CLASS_NAMES, FIGURES_DIR

def run_phase2():
    print("\n" + "="*70)
    print(" 📊 PHASE 2: DATA UNDERSTANDING & EDA")
    print("="*70)

    image_paths = []
    labels = []
    sizes = []
    channels = []
    corrupted_count = 0

    for cls in CLASS_NAMES:
        cls_dir = RAW_DATA_DIR / cls
        files = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg"))
        
        for f in files:
            try:
                with Image.open(f) as img:
                    img.verify() # Test image integrity
                
                with Image.open(f) as img:
                    w, h = img.size
                    c = len(img.getbands())
                    sizes.append((w, h))
                    channels.append(c)
                    image_paths.append(f)
                    labels.append(cls)
            except Exception as e:
                print(f"⚠️ Corrupted image detected: {f.name} ({e})")
                corrupted_count += 1

    print(f"\n🔍 EDA Inspection Findings:")
    print(f"   Total Valid Images: {len(image_paths)}")
    print(f"   Corrupted/Invalid Images: {corrupted_count}")

    if sizes:
        widths, heights = zip(*sizes)
        print(f"   Image Width Range: {min(widths)}px to {max(widths)}px (Avg: {np.mean(widths):.1f}px)")
        print(f"   Image Height Range: {min(heights)}px to {max(heights)}px (Avg: {np.mean(heights):.1f}px)")
        print(f"   Channels Detected: {set(channels)} (Grayscale/RGB)")

    # Plotting EDA Dashboard
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Phase 2: Exploratory Data Analysis (Medical Chest X-Ray)", fontsize=16, fontweight='bold')

    # Plot 1: Class Distribution Bar Chart
    unique_cls, counts = np.unique(labels, return_counts=True)
    sns.barplot(x=unique_cls, y=counts, ax=axes[0, 0], palette="viridis")
    axes[0, 0].set_title("Class Distribution (Imbalance Audit)")
    axes[0, 0].set_ylabel("Number of Images")
    for i, v in enumerate(counts):
        axes[0, 0].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

    # Plot 2: Image Aspect Ratio / Resolution Scatter
    if sizes:
        axes[0, 1].scatter(widths, heights, alpha=0.6, color='teal')
        axes[0, 1].set_title("Image Dimensions (Width vs Height)")
        axes[0, 1].set_xlabel("Width (px)")
        axes[0, 1].set_ylabel("Height (px)")

    # Plot 3: Pixel Intensity Histogram (Sample image per class)
    for cls in CLASS_NAMES:
        cls_files = [p for p, l in zip(image_paths, labels) if l == cls]
        if cls_files:
            img = np.array(Image.open(cls_files[0]).convert("L"))
            axes[0, 2].hist(img.ravel(), bins=30, alpha=0.5, label=cls, density=True)
    axes[0, 2].set_title("Pixel Intensity Distribution")
    axes[0, 2].set_xlabel("Pixel Value (0-255)")
    axes[0, 2].set_ylabel("Density")
    axes[0, 2].legend()

    # Plots 4-6: Visual Sample Images for each class
    for idx, cls in enumerate(CLASS_NAMES):
        cls_files = [p for p, l in zip(image_paths, labels) if l == cls]
        ax = axes[1, idx]
        if cls_files:
            img = Image.open(cls_files[0])
            ax.imshow(img, cmap='bone' if len(img.getbands())==1 else None)
            ax.set_title(f"Sample: {cls}")
        ax.axis('off')

    plt.tight_layout()
    output_fig_path = FIGURES_DIR / "phase2_eda_summary.png"
    plt.savefig(output_fig_path, dpi=300)
    plt.close()

    print(f"✅ Saved EDA Summary Chart: {output_fig_path}")
    print("✅ Phase 2 Complete!\n")
    return {"valid_images": len(image_paths), "corrupted": corrupted_count}

if __name__ == "__main__":
    run_phase2()
