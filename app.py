import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import pandas as pd

st.set_page_config(page_title="Rail Defect Detector", layout="centered")

st.title("🚆 Rail Surface Defect Detection")
st.write("Upload a rail surface image to detect type-1 and type-2 defects.")

# Load model once (cached so it doesn't reload on every interaction)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

def preprocess_grayscale_clahe(pil_img):
    """Converts external color images to grayscale + CLAHE enhancement,
    matching the training domain, without retraining."""
    img_array = np.array(pil_img.convert("L"))  # grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_array)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(enhanced_rgb)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

apply_preprocessing = st.checkbox(
    "Apply grayscale + contrast enhancement (recommended for color/phone photos)",
    value=True
)

confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    if apply_preprocessing:
        processed_image = preprocess_grayscale_clahe(image)
    else:
        processed_image = image

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        st.image(processed_image, caption="Preprocessed", use_container_width=True)

    with st.spinner("Detecting defects..."):
        results = model.predict(processed_image, conf=confidence_threshold, verbose=False)[0]

    # draw boxes on the image
    result_img = results.plot()  # returns numpy array (BGR)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    st.subheader("Detection Result")
    st.image(result_img, use_container_width=True)

    # build results table
    class_names = model.names
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append({
            "Class": class_names[cls_id],
            "Confidence": round(conf, 3)
        })

    if detections:
        df = pd.DataFrame(detections)
        st.subheader("Detected Defects")
        st.dataframe(df, use_container_width=True)

        st.subheader("Summary")
        summary = df["Class"].value_counts().reset_index()
        summary.columns = ["Defect Type", "Count"]
        st.table(summary)
    else:
        st.info("No defects detected above the confidence threshold.")