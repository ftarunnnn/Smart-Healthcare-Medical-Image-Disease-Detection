"""
Phase 9: Confidence Score & Prediction + Grad-CAM Explainability
------------------------------------------------------------------
- Performs single-image disease prediction with Softmax confidence probabilities.
- Computes Grad-CAM (Gradient-weighted Class Activation Mapping) for ROI explainability.
- Formulates decision-support framing with mandatory medical disclaimer.
- Generates Grad-CAM diagnostic plot -> outputs/figures/phase9_gradcam_prediction.png.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms, models
from config import MODELS_DIR, FIGURES_DIR, CLASS_NAMES, NUM_CLASSES, MEDICAL_DISCLAIMER

class GradCAM:
    """
    Grad-CAM implementation for PyTorch ResNet layer4 feature maps.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        return cam, target_class


def predict_medical_image(image_path_or_pil, model_path=None):
    """
    Given an image path or PIL Image, predicts disease class, confidence score,
    and returns Grad-CAM heatmap overlay.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ResNet model
    if model_path is None:
        model_path = MODELS_DIR / "resnet_model.pth"

    model = models.resnet18()
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, NUM_CLASSES))

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model = model.to(device)
    model.eval()

    # Image Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if isinstance(image_path_or_pil, (str, os.PathLike)):
        raw_pil = Image.open(image_path_or_pil).convert("RGB").resize((224, 224))
    else:
        raw_pil = image_path_or_pil.convert("RGB").resize((224, 224))

    input_tensor = transform(raw_pil).unsqueeze(0).to(device)

    # 1. Softmax Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

    pred_class = CLASS_NAMES[pred_idx]
    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    # 2. Grad-CAM Generation
    grad_cam = GradCAM(model, model.layer4[-1])
    cam, _ = grad_cam.generate(input_tensor, target_class=pred_idx)
    
    # Overlay heatmap onto raw image
    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    cam_np = np.array(cam_resized) / 255.0

    heatmap = cm.jet(cam_np)[:, :, :3]
    raw_np = np.array(raw_pil) / 255.0
    overlay = (0.6 * raw_np + 0.4 * heatmap)
    overlay = np.clip(overlay, 0, 1)

    viva_summary = (
        f"The system predicts the disease class as '{pred_class}' from the medical image "
        f"with a confidence probability score of {confidence*100:.1f}%.\n"
        f"It is intended strictly as a decision-support system, not a replacement for a medical professional."
    )

    return {
        "pred_class": pred_class,
        "confidence": confidence,
        "prob_dict": prob_dict,
        "raw_image": raw_pil,
        "heatmap": heatmap,
        "overlay": overlay,
        "viva_summary": viva_summary
    }


def run_phase9(sample_image_path=None):
    print("\n" + "="*70)
    print(" 🎯 PHASE 9: CONFIDENCE SCORE & PREDICTION + GRAD-CAM EXPLAINABILITY")
    print("="*70)

    if sample_image_path is None or not os.path.exists(sample_image_path):
        # Pick first sample image from raw dataset
        from config import RAW_DATA_DIR
        sample_files = list(RAW_DATA_DIR.rglob("*.png"))
        if sample_files:
            sample_image_path = sample_files[0]
        else:
            print("⚠️ No sample image found to test inference.")
            return None

    print(f"   Testing Inference on Sample Image: {sample_image_path.name}")
    res = predict_medical_image(sample_image_path)

    print("\n📈 Prediction Results:")
    print(f"   Predicted Disease Class : {res['pred_class'].upper()}")
    print(f"   Confidence Score        : {res['confidence']*100:.2f}%")
    print("\n   Class Probability Distribution:")
    for cls_name, p in res["prob_dict"].items():
        print(f"   - {cls_name:<15}: {p*100:.2f}%")

    print("\n💬 Viva Defense Framing:")
    print(f"   \"{res['viva_summary']}\"")

    # Save Grad-CAM Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(res["raw_image"])
    axes[0].set_title("Input X-Ray Image")
    axes[0].axis("off")

    axes[1].imshow(res["heatmap"])
    axes[1].set_title(f"Grad-CAM Heatmap ({res['pred_class']})")
    axes[1].axis("off")

    axes[2].imshow(res["overlay"])
    axes[2].set_title(f"Diagnostic ROI Overlay ({res['confidence']*100:.1f}%)")
    axes[2].axis("off")

    plt.suptitle("Phase 9: Medical Image Inference & Grad-CAM Heatmap", fontsize=14, fontweight="bold")
    out_fig = FIGURES_DIR / "phase9_gradcam_prediction.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()

    print(f"\n✅ Saved Grad-CAM Diagnostic Chart: {out_fig}")
    print("✅ Phase 9 Complete!\n")
    return res

if __name__ == "__main__":
    run_phase9()
