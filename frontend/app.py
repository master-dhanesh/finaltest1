import os
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Fresh vs Rotten", page_icon="🍎")
st.title("🍎 Fresh vs Rotten Image Classifier")

API_URL = os.getenv(
    "API_URL", "http://localhost:8000"
)  # change in Streamlit Cloud secrets later

st.write(
    "Upload an image and the app will call the **FastAPI** backend to predict **Fresh** or **Rotten**."
)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    if st.button("Predict"):
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        with st.spinner("Calling API..."):
            resp = requests.post(f"{API_URL}/predict", files=files, timeout=60)

        if resp.status_code != 200:
            st.error(f"API error: {resp.status_code}\n{resp.text}")
        else:
            data = resp.json()
            label = data.get("label", "Unknown")
            conf = data.get("confidence", None)

            st.subheader(f"Prediction: {label}")
            if conf is not None:
                st.write(f"Confidence: **{conf:.4f}**")
            st.json(data)
