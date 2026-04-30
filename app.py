import streamlit as st
import numpy as np
import cv2
import h5py
import os
import joblib
import gdown

from utils import extract_features

# =========================
# GOOGLE DRIVE FILE IDS
# =========================
MODEL_FILE_ID = "YOUR_MODEL_FILE_ID"
SCALER_FILE_ID = "YOUR_SCALER_FILE_ID"
IMPUTER_FILE_ID = "YOUR_IMPUTER_FILE_ID"

# =========================
# DOWNLOAD FUNCTION
# =========================
def download_file(file_id, output_name):
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output_name):
        gdown.download(url, output_name, quiet=False)

# =========================
# DOWNLOAD MODELS
# =========================
download_file(MODEL_FILE_ID, "model.pkl")
download_file(SCALER_FILE_ID, "scaler.pkl")
download_file(IMPUTER_FILE_ID, "imputer.pkl")

# =========================
# LOAD MODELS
# =========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

# =========================
# UI
# =========================
st.set_page_config(page_title="Brain Tumor AI", layout="centered")

st.title("🧠 Brain Tumor Classification System")
st.write("Upload MRI image (.mat or image file)")

# =========================
# FILE UPLOAD
# =========================
file = st.file_uploader("Upload File")

if file:

    # ---- Load image ----
    if file.name.endswith(".mat"):
        with h5py.File(file, 'r') as f:
            img = np.array(f['cjdata']['image']).T
    else:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 0)

    img = cv2.resize(img, (256, 256))
    st.image(img, caption="Input MRI")

    # ---- Feature extraction ----
    features = extract_features(img)
    X = np.array(features).reshape(1, -1)

    # ---- preprocessing ----
    X = imputer.transform(X)
    X = scaler.transform(X)

    # ---- prediction ----
    if st.button("Predict"):
        pred = model.predict(X)
        st.success(f"Prediction Class: {pred[0]}")
