# freshness_predictor.py
# Robust Keras + Google Drive auto-download version

import os
import traceback
import numpy as np
import pickle
from PIL import Image
import gdown

# -------------------------------------------------
# Paths
# -------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

IMG_SIZE = (150, 150)

# -------------------------------------------------
# Google Drive File IDs
# -------------------------------------------------
MODEL_IDS = {
    "fruit_model_retrained.h5": "1yOhNvH71E0fZqrbXnx0LL23qbIhPDP5T",
    "fruit_model.h5": "1RSPBmSpx425vvSloWwX24fGUNFlx9927",
    "label_encoder_v5.pkl": "1NKJjROuaQ9fl7wj69dhWybfR4jCxAg7L",
    "label_encoder_v2.pkl": "1QJj6sV7_eh0cxzWziLZQh9YXHSCjOsd5",
}

MODEL_PATHS = [
    os.path.join(MODELS_DIR, "fruit_model_retrained.h5"),
    os.path.join(MODELS_DIR, "fruit_model.h5"),
]

LABEL_ENCODER_PATHS = [
    os.path.join(MODELS_DIR, "label_encoder_v5.pkl"),
    os.path.join(MODELS_DIR, "label_encoder_v2.pkl"),
    os.path.join(MODELS_DIR, "label_encoder.pkl"),
]

# -------------------------------------------------
# Auto-download models
# -------------------------------------------------
def download_if_missing():
    os.makedirs(MODELS_DIR, exist_ok=True)

    for filename, file_id in MODEL_IDS.items():
        file_path = os.path.join(MODELS_DIR, filename)

        if not os.path.exists(file_path):
            print(f"[predictor] Downloading {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            try:
                gdown.download(url, file_path, quiet=False)
                print(f"[predictor] Downloaded: {filename}")
            except Exception as e:
                print(f"[predictor] Failed to download {filename}:", e)

# -------------------------------------------------
# Load Keras
# -------------------------------------------------
KERAS_OK = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    KERAS_OK = True
except Exception:
    KERAS_OK = False

def load_any_keras_model(paths):
    if not KERAS_OK:
        print("[predictor] TensorFlow not available.")
        return None

    for p in paths:
        if os.path.exists(p):
            try:
                model = load_model(p)
                print("[predictor] Loaded keras model:", p)
                return model
            except Exception as e:
                print("[predictor] Failed loading model:", e)

    return None

def load_any_label_encoder(paths):
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    le = pickle.load(f)
                print("[predictor] Loaded label encoder:", p)
                return le
            except Exception as e:
                print("[predictor] Failed loading label encoder:", e)

    return None

# -------------------------------------------------
# Initialize
# -------------------------------------------------
download_if_missing()

keras_model = load_any_keras_model(MODEL_PATHS)
label_encoder = load_any_label_encoder(LABEL_ENCODER_PATHS)

# -------------------------------------------------
# Prediction Function
# -------------------------------------------------
def predict_freshness_from_pil(pil_img):
    """
    Returns:
    {
        "label": str,
        "confidence": float,
        "source": str,
        "debug": dict
    }
    """

    try:
        img = pil_img.convert("RGB").resize(IMG_SIZE)
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, 0)

        debug = {}

        if keras_model is None:
            return {
                "label": "Unsure",
                "confidence": 0.0,
                "source": "no_model",
                "debug": {"error": "Model not loaded"},
            }

        pred = keras_model.predict(arr)
        pred = np.array(pred)
        debug["raw_prediction"] = pred.tolist()

        # ------------------------------
        # Binary classification
        # ------------------------------
        if pred.ndim == 1 or (pred.ndim == 2 and pred.shape[-1] == 1):
            prob = float(pred.reshape(-1)[0])
            confidence = round(prob * 100, 2)

            if prob >= 0.5:
                label = "OK to eat"
            else:
                label = "Not OK"

            return {
                "label": label,
                "confidence": confidence,
                "source": "keras_binary",
                "debug": debug,
            }

        # ------------------------------
        # Multi-class classification
        # ------------------------------
        pred_vec = pred.reshape(-1)
        pred_idx = int(np.argmax(pred_vec))
        confidence = float(pred_vec[pred_idx]) * 100
        confidence = round(confidence, 2)

        predicted_class = ""
        if label_encoder is not None:
            try:
                predicted_class = label_encoder.inverse_transform([pred_idx])[0]
                debug["predicted_class"] = predicted_class
            except Exception:
                pass

        predicted_class_lower = predicted_class.lower()

        if "fresh" in predicted_class_lower:
            label = "OK to eat"
        elif "rotten" in predicted_class_lower:
            label = "Not OK"
        else:
            # fallback if class names are weird
            if confidence >= 60:
                label = "OK to eat"
            elif confidence <= 40:
                label = "Not OK"
            else:
                label = "Unsure"

        return {
            "label": label,
            "confidence": confidence,
            "source": "keras_multiclass",
            "debug": debug,
        }

    except Exception as e:
        return {
            "label": "Unsure",
            "confidence": 0.0,
            "source": "error",
            "debug": {
                "error": str(e),
                "trace": traceback.format_exc(),
            },
        }
