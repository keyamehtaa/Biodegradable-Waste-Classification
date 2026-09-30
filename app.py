import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Biodegradable Classifier", page_icon="♻️", layout="centered")

@st.cache_resource
def load_model(model_path="best.pt"):
    return YOLO(model_path)

model = load_model()

st.title("♻️ Biodegradable vs Non-Biodegradable Classifier")
st.write("Upload an image and the model will predict its class with a confidence score.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Predicting..."):
        results = model.predict(source=np.array(image), verbose=False)

    result = results[0]  

    if hasattr(result, "probs") and result.probs is not None:
        probs = result.probs
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        class_name = model.names[top1_idx]

        st.subheader("Prediction")
        st.markdown(f"**Class:** `{class_name}`")
        st.markdown(f"**Confidence:** `{top1_conf * 100:.2f}%`")
        st.progress(top1_conf)

        # Show all class probabilities
        st.subheader("Class Probabilities")
        all_probs = probs.data.tolist()
        for idx, p in enumerate(all_probs):
            st.write(f"{model.names[idx]}: {p * 100:.2f}%")
    else:
        st.error("This model doesn't appear to be a classification model (no `.probs` found). Make sure you loaded a YOLO **-cls** model.")
else:
    st.info("Please upload an image to get started.")
