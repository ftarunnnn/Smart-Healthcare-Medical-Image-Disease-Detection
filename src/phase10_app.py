"""
Phase 10: Deployment & User Interface (Streamlit Web Dashboard)
--------------------------------------------------------------
- Sleek, modern dark-themed interactive Streamlit Web Dashboard.
- Live X-ray image drag-and-drop uploader with softmax confidence scoring.
- Grad-CAM heatmap visualization for diagnostic explainability.
- Phase-by-Phase Viva Sandbox showing outputs from all 10 project phases.
- Prominent Medical Decision-Support Disclaimer.
"""

import os
import sys
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Add root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from config import CLASS_NAMES, FIGURES_DIR, MEDICAL_DISCLAIMER, REPORTS_DIR
from src.phase9_confidence_prediction import predict_medical_image

# Page Setup
st.set_page_config(
    page_title="Smart Healthcare - Medical Image Disease Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Dark Theme Customization */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2640 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .disclaimer-box {
        background-color: #3b1419;
        border-left: 5px solid #ef4444;
        color: #fca5a5;
        padding: 15px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-size: 0.95rem;
    }
    .viva-box {
        background-color: #132a1e;
        border-left: 5px solid #10b981;
        color: #6ee7b7;
        padding: 15px 20px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    h1, h2, h3 {
        color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/isometric/100/stethoscope.png", width=70)
st.sidebar.title("Smart Healthcare AI")
st.sidebar.caption("Medical Image Disease Detection System")

app_mode = st.sidebar.radio(
    "Navigation Menu",
    [
        "🩺 Live Image Diagnosis & Grad-CAM",
        "🎓 Phase-by-Phase Viva Sandbox",
        "📚 Viva Exam Cheatsheet & Q&A"
    ]
)

# Main Banner Disclaimer
st.markdown(f'<div class="disclaimer-box">{MEDICAL_DISCLAIMER}</div>', unsafe_allow_html=True)


if app_mode == "🩺 Live Image Diagnosis & Grad-CAM":
    st.title("🩺 Medical Image Inference & Decision Support")
    st.write("Upload a chest X-ray image to compute disease classification probabilities, confidence scores, and Grad-CAM region of interest (ROI) heatmaps.")

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("1. Select or Upload X-Ray Image")
        uploaded_file = st.file_uploader("Upload Chest X-Ray (PNG / JPG / JPEG)", type=["png", "jpg", "jpeg"])
        
        # Sample selection fallback
        use_sample = st.checkbox("Or use a pre-loaded Sample Image", value=False if uploaded_file else True)
        
        input_image = None
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
        elif use_sample:
            from config import RAW_DATA_DIR
            sample_files = list(RAW_DATA_DIR.rglob("*.png"))
            if sample_files:
                sample_path = st.selectbox("Choose Sample X-Ray:", [f.name for f in sample_files])
                chosen_file = [f for f in sample_files if f.name == sample_path][0]
                input_image = Image.open(chosen_file)

        if input_image:
            st.image(input_image, caption="Target Medical Image", use_container_width=True)

    with col_right:
        if input_image:
            st.subheader("2. AI Diagnostic Prediction & Confidence")
            
            with st.spinner("Processing image through PyTorch ResNet & Grad-CAM hooks..."):
                res = predict_medical_image(input_image)

            pred_class = res["pred_class"]
            conf = res["confidence"] * 100

            # Badge color
            color_map = {"Normal": "#10b981", "Pneumonia": "#ef4444", "Tuberculosis": "#f59e0b"}
            badge_color = color_map.get(pred_class, "#3b82f6")

            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin:0; color:#94a3b8;">Predicted Disease Class</h4>
                <h1 style="color:{badge_color} !important; font-size: 2.5rem; margin:10px 0;">{pred_class}</h1>
                <p style="font-size:1.2rem; color:#cbd5e1; margin:0;">Confidence Score: <b>{conf:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            # Viva Framing Banner
            st.markdown(f"""
            <div class="viva-box">
                <b>💡 Viva Decision-Support Framing:</b><br>
                "{res['viva_summary']}"
            </div>
            """, unsafe_allow_html=True)

            # Probability Gauge Chart
            st.subheader("3. Probability Breakdown")
            prob_df = res["prob_dict"]
            fig_prob = px.bar(
                x=list(prob_df.keys()),
                y=[v * 100 for v in prob_df.values()],
                labels={'x': 'Disease Class', 'y': 'Probability (%)'},
                color=list(prob_df.keys()),
                color_discrete_map=color_map,
                text_auto='.1f'
            )
            fig_prob.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e6ed'),
                yaxis=dict(range=[0, 105])
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # Grad-CAM Heatmap
            st.subheader("4. Explainable AI: Grad-CAM Diagnostic Heatmap")
            st.write("Highlighted warm regions (red/yellow) indicate where the deep learning model focused its attention to determine the disease classification.")
            
            cam_col1, cam_col2 = st.columns(2)
            with cam_col1:
                st.image(res["heatmap"], caption="Grad-CAM Activation Heatmap", use_container_width=True)
            with cam_col2:
                st.image(res["overlay"], caption="Diagnostic ROI Overlay on X-Ray", use_container_width=True)


elif app_mode == "🎓 Phase-by-Phase Viva Sandbox":
    st.title("🎓 Phase-by-Phase Viva Execution Sandbox")
    st.write("Browse visual outputs, charts, and technical summaries produced across all 10 project phases.")

    phase_tabs = st.tabs([
        "Phase 1 & 2: EDA",
        "Phase 3 & 4: Preprocess & Augment",
        "Phase 5: Split",
        "Phase 6: ResNet Training",
        "Phase 7: Feature Extraction + ML",
        "Phase 8: Evaluation & ROC",
        "Phase 9: Grad-CAM Explainability"
    ])

    with phase_tabs[0]:
        st.header("Phase 1 & 2: Dataset Collection & EDA Audit")
        st.markdown("""
        - **Objective**: Define disease targets and verify dataset integrity.
        - **EDA Tasks**: Checked class distribution, resolution variations, aspect ratios, corrupted image checks.
        """)
        eda_fig = FIGURES_DIR / "phase2_eda_summary.png"
        if eda_fig.exists():
            st.image(str(eda_fig), use_container_width=True)

    with phase_tabs[1]:
        st.header("Phase 3 & 4: Image Preprocessing & Augmentation")
        st.markdown("""
        - **Phase 3 (Preprocessing)**: Resized to 224x224, normalized to [0,1], applied **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for X-ray contrast enhancement.
        - **Phase 4 (Augmentation)**: Applied rotation (±10°), zoom, horizontal flip. Avoided vertical flip to preserve top-down anatomical orientation.
        """)
        c1, c2 = st.columns(2)
        p3_fig = FIGURES_DIR / "phase3_preprocessing_comparison.png"
        p4_fig = FIGURES_DIR / "phase4_augmentation_grid.png"
        if p3_fig.exists():
            c1.image(str(p3_fig), caption="Phase 3: CLAHE Preprocessing", use_container_width=True)
        if p4_fig.exists():
            c2.image(str(p4_fig), caption="Phase 4: Data Augmentation Grid", use_container_width=True)

    with phase_tabs[2]:
        st.header("Phase 5: Stratified Train-Val-Test Split")
        st.markdown("""
        - **Ratio**: 70% Training, 15% Validation, 15% Testing.
        - **Data Leakage Prevention**: Split performed BEFORE data augmentation so validation/test sets remain completely untouched.
        """)
        p5_fig = FIGURES_DIR / "phase5_split_distribution.png"
        if p5_fig.exists():
            st.image(str(p5_fig), use_container_width=True)

    with phase_tabs[3]:
        st.header("Phase 6: Transfer Learning (ResNet18 / Custom CNN)")
        st.markdown("""
        - **Architecture**: PyTorch ResNet18 backbone pretrained on ImageNet, fine-tuned on medical X-rays.
        - **Optimization**: Adam Optimizer ($lr=0.001$), CrossEntropy Loss.
        """)
        p6_fig = FIGURES_DIR / "phase6_learning_curves.png"
        if p6_fig.exists():
            st.image(str(p6_fig), use_container_width=True)

    with phase_tabs[4]:
        st.header("Phase 7: Feature Extraction + Classical ML Comparison")
        st.markdown("""
        - **Bottleneck Features**: Extracted 512-dimensional feature vectors from ResNet average pooling layer.
        - **Classifiers**: Support Vector Machine (SVM RBF), Random Forest, XGBoost.
        """)
        p7_fig = FIGURES_DIR / "phase7_ml_classifier_comparison.png"
        if p7_fig.exists():
            st.image(str(p7_fig), use_container_width=True)

    with phase_tabs[5]:
        st.header("Phase 8: Model Evaluation Metrics")
        st.markdown("""
        - **Metrics**: Accuracy, Precision, Recall (Sensitivity), F1-Score, Confusion Matrix, and ROC-AUC curves.
        """)
        c1, c2 = st.columns(2)
        p8_cm = FIGURES_DIR / "phase8_confusion_matrix.png"
        p8_roc = FIGURES_DIR / "phase8_roc_auc_curve.png"
        if p8_cm.exists():
            c1.image(str(p8_cm), caption="Confusion Matrix", use_container_width=True)
        if p8_roc.exists():
            c2.image(str(p8_roc), caption="Multi-Class ROC-AUC Curves", use_container_width=True)

        report_txt = REPORTS_DIR / "phase8_evaluation_report.txt"
        if report_txt.exists():
            with st.expander("📄 View Full Classification Report Text"):
                st.code(report_txt.read_text(), language="text")

    with phase_tabs[6]:
        st.header("Phase 9: Grad-CAM Explainable AI")
        st.markdown("""
        - **Gradient-weighted Class Activation Mapping**: Visualizes deep convolutional layer activation gradients to highlight diagnostic regions of interest (ROI).
        """)
        p9_fig = FIGURES_DIR / "phase9_gradcam_prediction.png"
        if p9_fig.exists():
            st.image(str(p9_fig), use_container_width=True)


elif app_mode == "📚 Viva Exam Cheatsheet & Q&A":
    st.title("📚 Viva Defense Cheatsheet & Key Q&As")
    st.write("Essential answers and technical talking points for defending this project in academic viva examinations.")

    st.markdown("""
    ### 🔑 The #1 Most Important Viva Rule
    > **❌ Never say**: "The AI model confirms that the patient has Pneumonia."
    > 
    > **✅ Always say**: "The system predicts the disease class from the medical image and provides a confidence probability score. It is designed as a **decision-support assistant** for radiologists, not a replacement for a medical professional."

    ---
    ### ❓ Top 6 Viva Questions & Concise Model Answers

    #### Q1: Why use Transfer Learning (ResNet) instead of training a CNN from scratch?
    - **Answer**: Medical image datasets are often small. Training deep neural networks from scratch risks severe overfitting. Transfer learning leverages rich visual feature representations (edges, textures, shapes) pre-trained on ImageNet, fine-tuning only the classification head for fast convergence and high generalization.

    #### Q2: How did you prevent Data Leakage?
    - **Answer**: We performed the Stratified Train-Val-Test split **before** applying data augmentation. If augmentation is applied before splitting, augmented variants of the same image can end up in both training and test sets, artificially inflating test accuracy.

    #### Q3: Why is CLAHE (Contrast Limited Adaptive Histogram Equalization) useful for X-rays?
    - **Answer**: Standard X-rays often suffer from low contrast in soft-tissue lung areas. CLAHE enhances local contrast while limiting noise amplification, making subtle pulmonary infiltrates clearer for feature extraction.

    #### Q4: Why is vertical flipping excluded during Data Augmentation?
    - **Answer**: Anatomical structure has a fixed top-down orientation (clavicles at the top, diaphragm at the bottom). Vertical flipping creates anatomically unrealistic images that disrupt model learning.

    #### Q5: What is Grad-CAM and why is it important in healthcare AI?
    - **Answer**: Deep learning models are often viewed as "black boxes." Grad-CAM computes gradients of the target class score with respect to the final convolutional feature maps, creating a visual heatmap overlay that shows radiologists *which region of the X-ray* influenced the prediction.

    #### Q6: Why compare Deep Learning with Feature Extraction + SVM/XGBoost?
    - **Answer**: It proves architectural rigor. ResNet serves as a feature extractor (512-dim embedding), while classical classifiers (SVM, XGBoost) provide interpretable, lightweight decision boundaries for benchmark comparison.
    """)
