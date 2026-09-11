import os
import sys
import io
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Data Directories
DATA_DIR = BASE_DIR / "dataset"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLIT_DATA_DIR = DATA_DIR / "splits"

# Output Directories
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Ensure directories exist
for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLIT_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Image Configuration
IMAGE_SIZE = (224, 224)
CHANNELS = 3
CLASS_NAMES = ["Normal", "Pneumonia", "Tuberculosis"]
NUM_CLASSES = len(CLASS_NAMES)

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 5  # Quick training for viva demonstration
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# Medical Viva Disclaimer
MEDICAL_DISCLAIMER = (
    "⚠️ MEDICAL DECISION-SUPPORT DISCLAIMER:\n"
    "This system provides disease class predictions with confidence probability scores intended "
    "strictly as a decision-support assistant for medical professionals. "
    "It is NOT a replacement for a certified radiologist or clinical medical diagnosis."
)
