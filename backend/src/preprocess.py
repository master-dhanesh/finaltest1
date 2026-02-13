import numpy as np
from PIL import Image


def preprocess_image(image: Image.Image, size=(224, 224)) -> np.ndarray:
    img = image.convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, C)
    return arr
