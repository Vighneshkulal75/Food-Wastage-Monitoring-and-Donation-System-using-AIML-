# freshness_predictor.py
# Clean & stable version

import os
import traceback
import numpy as np
import pickle
from PIL import Image
import gdown

# ---------------- GOOGLE DRIVE MODEL IDS ----------------
MODEL_IDS = {
    "fruit_model_retrained.h5": "1yOhNvH71E0fZqrbXnx0LL23qbIhPDP5T",
    "fruit_model.h5": "1RSPBmSpx425vvSloWwX24fGUNFlx9927",
    "label_encoder_v2.pkl": "1QJj6sV7_eh0cxzWziLZQh9YXHSCjOsd5",
}

# ---------------- PATH SETUP ----------------
THIS_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "fruit_model_retrained.h5")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder_v2.pkl")

IMG_SIZE = (150, 150)

# ---------------- DOWNLOAD MODELS ----------------
def download_if_missing():
    os.makedirs(MODELS_DIR, exist_ok=True)

    for filename, file_id in MODEL_IDS.items():
        file_path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(file_path):
            print(f"[predictor] Downloading {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, file_path, quiet=False)

download_if_missing()

# ---------------- LOAD MODEL ----------------
try:
    from tensorflow.keras.models import load_model
    keras_model = load_model(MODEL_PATH)
    print("[predictor] Loaded model")
except Exception as e:
    keras_model = None
    print("[predictor] Model load failed:", e)

# ---------------- LOAD LABEL ENCODER ----------------
try:
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    print("[predictor] Loaded label encoder")
except Exception as e:
    label_encoder = None
    print("[predictor] Label encoder load failed:", e)


# =====================================================
#                 MAIN FUNCTION
# =====================================================

def predict_freshness_from_pil(pil_img):
    try:
        if keras_model is None or label_encoder is None:
            return {
                "label": "Unsure",
                "confidence": 0.0,
                "source": "model_missing",
                "debug": {}
            }

        # Preprocess
        img = pil_img.convert("RGB").resize(IMG_SIZE)
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, 0)

        # Predict
        preds = keras_model.predict(arr)[0]
        preds = np.array(preds)

        predicted_index = int(np.argmax(preds))
        confidence = float(preds[predicted_index]) * 100

        predicted_class = label_encoder.inverse_transform([predicted_index])[0]
        predicted_class_lower = predicted_class.lower().strip()

        print("Predicted class:", predicted_class)
        print("Confidence:", confidence)

        # -------- SAFE MAPPING --------
        if "rotten" in predicted_class_lower:
            final_label = "Not OK"
        elif "fresh" in predicted_class_lower:
            final_label = "OK to eat"
        else:
            final_label = "Unsure"

        return {
            "label": final_label,
            "confidence": round(confidence, 2),
            "source": "keras_argmax",
            "debug": {
                "raw_class": predicted_class
            }
        }

    except Exception as e:
        return {
            "label": "Unsure",
            "confidence": 0.0,
            "source": "error",
            "debug": {"error": str(e), "trace": traceback.format_exc()},
        }
