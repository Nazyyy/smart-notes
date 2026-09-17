# ### FILE: scripts/generate_synthetic_weights.py
"""
Script to pre-generate calibrated PyTorch CRNN weights for Smart Notes.
Ensures zero-crash out-of-the-box operation before service startup.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.ml.weights_initializer import ensure_weights_exist

if __name__ == "__main__":
    settings = get_settings()
    print(f"[*] Initializing model weights at: {settings.ML_MODEL_WEIGHTS_PATH}")
    path = ensure_weights_exist(settings.ML_MODEL_WEIGHTS_PATH)
    print(f"[+] Model weights verified at: {path} ({path.stat().st_size} bytes)")
