"""
End-to-End Pipeline Execution Script
------------------------------------
Executes Phases 1 through 9 sequentially:
Phase 1: Dataset Collection & Verification
Phase 2: Exploratory Data Analysis (EDA)
Phase 3: Image Preprocessing (CLAHE)
Phase 4: Data Augmentation
Phase 5: Stratified Train-Val-Test Split
Phase 6: ResNet Transfer Learning Training
Phase 7: Feature Extraction + Classical ML (SVM, RF, XGBoost)
Phase 8: Comprehensive Model Evaluation (ROC-AUC & Confusion Matrix)
Phase 9: Confidence Prediction & Grad-CAM Heatmap
"""

import time
from src.phase1_dataset import run_phase1
from src.phase2_eda import run_phase2
from src.phase3_preprocessing import run_phase3
from src.phase4_augmentation import run_phase4
from src.phase5_split import run_phase5
from src.phase6_cnn_transfer import run_phase6
from src.phase7_feature_extraction_ml import run_phase7
from src.phase8_evaluation import run_phase8
from src.phase9_confidence_prediction import run_phase9

def main():
    start_time = time.time()
    print("="*80)
    print(" SMART HEALTHCARE - MEDICAL IMAGE DISEASE DETECTION PIPELINE")
    print("="*80)

    # Execute Phase 1
    run_phase1()

    # Execute Phase 2
    run_phase2()

    # Execute Phase 3
    run_phase3()

    # Execute Phase 4
    run_phase4()

    # Execute Phase 5
    run_phase5()

    # Execute Phase 6
    run_phase6(model_type="resnet18")

    # Execute Phase 7
    run_phase7()

    # Execute Phase 8
    run_phase8()

    # Execute Phase 9
    run_phase9()

    elapsed = time.time() - start_time
    print("="*80)
    print(f"PIPELINE EXECUTED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("="*80)
    print("\nNext Steps:")
    print("   To launch Phase 10 Interactive Streamlit Dashboard, run:")
    print("   streamlit run src/phase10_app.py\n")

if __name__ == "__main__":
    main()
