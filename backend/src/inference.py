import os
import numpy as np
import tensorflow as tf
from PIL import Image
from .preprocess import preprocess_image

# MODEL_PATH = os.getenv("MODEL_PATH", "../models/fruits_classification_model.keras")
MODEL_PATH = os.getenv("MODEL_PATH", "../models/latest_model.keras")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def predict(image: Image.Image) -> dict:
    model = get_model()
    x = preprocess_image(image)

    y = model.predict(x)
    y = np.array(y)

    # Handle common cases:
    # 1) Binary sigmoid output shape: (1,1)
    # 2) Softmax output shape: (1,n)
    if y.ndim == 2 and y.shape[1] == 1:
        score = float(y[0][0])  # sigmoid prob for "rotten" (as per your UI logic)
        label = "Rotten" if score > 0.5 else "Fresh"
        confidence = score if label == "Rotten" else 1.0 - score
        return {"label": label, "confidence": float(confidence), "raw_score": score}

    # Softmax / multi-class
    probs = y[0].astype(float)
    idx = int(np.argmax(probs))
    return {
        "label": str(idx),
        "confidence": float(probs[idx]),
        "raw_score": probs.tolist(),
    }
