import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from config import RAW_DATA_DIR, CLASS_NAMES, IMAGE_SIZE, RANDOM_SEED

def generate_chest_xray(class_name, seed=42):
    """
    Generates a synthetic chest X-ray image with anatomical lung shapes
    and class-specific visual markers (infiltrates / opacities / cavities).
    """
    rng = np.random.RandomState(seed)
    w, h = IMAGE_SIZE
    
    # 1. Base dark background (body chest cavity)
    img = Image.new("L", (w, h), color=25)
    draw = ImageDraw.Draw(img)
    
    # 2. Chest outline / Spine / Rib structures
    # Spine in center
    draw.rectangle([w//2 - 6, 20, w//2 + 6, h - 20], fill=120)
    
    # Ribs
    for y in range(40, h - 40, 20):
        # Left rib arc
        draw.arc([30, y, w//2 - 10, y + 30], start=180, end=360, fill=90, width=4)
        # Right rib arc
        draw.arc([w//2 + 10, y, w - 30, y + 30], start=180, end=360, fill=90, width=4)
    
    # 3. Left and Right Lung Fields (radiolucent = darker)
    lung_color = 50
    # Left Lung
    draw.ellipse([35, 45, w//2 - 15, h - 45], fill=lung_color)
    # Right Lung
    draw.ellipse([w//2 + 15, 45, w - 35, h - 45], fill=lung_color)
    
    # 4. Heart Shadow (middle-left radiopaque = brighter)
    draw.ellipse([w//2 - 35, h//2, w//2 + 10, h - 50], fill=140)
    
    # 5. Clavicles (Top collar bones)
    draw.line([30, 35, w//2 - 10, 45], fill=150, width=5)
    draw.line([w//2 + 10, 45, w - 30, 35], fill=150, width=5)

    # Convert to numpy array for noise and disease features
    img_arr = np.array(img, dtype=np.float32)
    
    # Add tissue grain noise
    grain = rng.normal(0, 8, img_arr.shape)
    img_arr = np.clip(img_arr + grain, 0, 255)

    # 6. Disease specific indicators
    if class_name == "Pneumonia":
        # Dense white cloudy opacities in lower lung field (consolidation)
        for _ in range(rng.randint(2, 5)):
            cx = rng.randint(45, w//2 - 25)
            cy = rng.randint(h//2, h - 60)
            radius = rng.randint(15, 30)
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            opacity = np.exp(-(dist**2)/(2 * (radius**2))) * rng.randint(80, 140)
            img_arr += opacity
            
    elif class_name == "Tuberculosis":
        # Focal apical (upper lung) cavitation nodules & patchy infiltrates
        for _ in range(rng.randint(2, 4)):
            cx = rng.randint(w//2 + 20, w - 45)
            cy = rng.randint(55, h//2 - 10)
            radius = rng.randint(10, 20)
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            opacity = np.exp(-(dist**2)/(2 * (radius**2))) * rng.randint(100, 160)
            img_arr += opacity

    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    final_img = Image.fromarray(img_arr).filter(ImageFilter.GaussianBlur(1.0))
    # Convert to RGB
    return final_img.convert("RGB")


def create_sample_dataset(samples_per_class=40):
    """
    Creates a synthetic Chest X-Ray dataset divided into folders by class.
    """
    print("Initializing Sample Dataset Generation...")
    np.random.seed(RANDOM_SEED)
    total_count = 0
    
    for cls in CLASS_NAMES:
        cls_dir = RAW_DATA_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(samples_per_class):
            img = generate_chest_xray(cls, seed=RANDOM_SEED + i + hash(cls)%1000)
            img_path = cls_dir / f"{cls.lower()}_sample_{i+1:03d}.png"
            img.save(img_path)
            total_count += 1
            
    print(f"Created {total_count} sample chest X-ray images in '{RAW_DATA_DIR}' ({samples_per_class} per class).")


if __name__ == "__main__":
    create_sample_dataset()
