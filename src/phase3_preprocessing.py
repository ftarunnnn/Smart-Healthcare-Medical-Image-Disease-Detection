"""
Phase 3: Image Preprocessing
----------------------------
- Resizes images to (224, 224).
- Normalizes pixel values to [0, 1].
- Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for X-Ray contrast enhancement.
- Stores output in dataset/processed.
- Generates Before vs After comparison visual chart.
"""

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CLASS_NAMES, IMAGE_SIZE, FIGURES_DIR

def apply_clahe(img_np):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for X-ray contrast enhancement.
    If opencv is unavailable, uses adaptive histogram stretch.
    """
    try:
        import cv2
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return enhanced_rgb
    except ImportError:
        # Fallback NumPy adaptive contrast stretch
        p2, p98 = np.percentile(img_np, (2, 98))
        img_rescaled = np.clip((img_np - p2) / (p98 - p2 + 1e-5) * 255.0, 0, 255).astype(np.uint8)
        return img_rescaled


def preprocess_image(img_path, target_size=IMAGE_SIZE):
    """
    Loads, resizes, enhances contrast, and normalizes an image.
    """
    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB").resize(target_size, Image.BILINEAR)
        img_np = np.array(img_rgb)
        
    enhanced_np = apply_clahe(img_np)
    normalized = enhanced_np.astype(np.float32) / 255.0
    return img_np, enhanced_np, normalized


def run_phase3():
    print("\n" + "="*70)
    print(" 🛠️ PHASE 3: IMAGE PREPROCESSING & CONTRAST ENHANCEMENT")
    print("="*70)
    
    total_processed = 0
    sample_before = None
    sample_after = None

    for cls in CLASS_NAMES:
        raw_cls_dir = RAW_DATA_DIR / cls
        proc_cls_dir = PROCESSED_DATA_DIR / cls
        proc_cls_dir.mkdir(parents=True, exist_ok=True)
        
        files = list(raw_cls_dir.glob("*.png")) + list(raw_cls_dir.glob("*.jpg")) + list(raw_cls_dir.glob("*.jpeg"))
        
        for f in files:
            orig, enhanced, norm = preprocess_image(f)
            
            # Save enhanced image
            out_img = Image.fromarray(enhanced)
            out_path = proc_cls_dir / f.name
            out_img.save(out_path)
            total_processed += 1
            
            if sample_before is None:
                sample_before = orig
                sample_after = enhanced

    print(f"\n⚙️ Preprocessing Operations Executed:")
    print(f"   1. Resized all images to {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} pixels.")
    print("   2. Applied CLAHE (Contrast Limited Adaptive Histogram Equalization).")
    print("   3. Pixel normalization scaled to range [0.0, 1.0].")
    print(f"   Saved {total_processed} preprocessed images to '{PROCESSED_DATA_DIR}'.")

    # Plot Before vs After Comparison
    if sample_before is not None and sample_after is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(sample_before)
        axes[0].set_title("Original Raw Image")
        axes[0].axis("off")
        
        axes[1].imshow(sample_after)
        axes[1].set_title("Preprocessed Image (CLAHE Enhanced)")
        axes[1].axis("off")
        
        plt.suptitle("Phase 3: Image Preprocessing Comparison", fontsize=14, fontweight="bold")
        out_fig = FIGURES_DIR / "phase3_preprocessing_comparison.png"
        plt.savefig(out_fig, dpi=300)
        plt.close()
        print(f"✅ Saved Comparison Chart: {out_fig}")

    print("✅ Phase 3 Complete!\n")
    return total_processed

if __name__ == "__main__":
    run_phase3()
