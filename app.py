import streamlit as st
import numpy as np
import cv2
import h5py
import os
import joblib
import gdown

from scipy.stats import skew, kurtosis
from statsmodels.stats.stattools import durbin_watson

# =========================
# GOOGLE DRIVE FILE IDS
# =========================
MODEL_FILE_ID = "1Cu8QVXtO-YTVMBidQuIKibEINuO8wcU_"
SCALER_FILE_ID = "1HFqf175O1Y5xhUyOV3wjkEtpsEeWrKPR"
IMPUTER_FILE_ID = "178PwZ-87uELP-3K1UNqX3HQ_pBNWWREx"

# =========================
# DOWNLOAD FUNCTION
# =========================
def download_file(file_id, output_name):
    if not os.path.exists(output_name):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
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
# FEATURE EXTRACTION
# =========================
def extract_features(img):
    p = img.flatten().astype(np.float64)
    p = np.nan_to_num(p)

    return [
        np.mean(p),
        np.std(p),
        np.var(p),
        np.min(p),
        np.max(p),
        np.median(p),
        skew(p),
        kurtosis(p),
        durbin_watson(p)
    ]

# =========================
# UI
# =========================
st.set_page_config(page_title="Brain Tumor AI", layout="centered")

st.title("🧠 Brain Tumor Classification System")
st.info("Upload only single MRI (.mat / image)")

# =========================
# UPLOAD
# =========================
file = st.file_uploader("Upload MRI File")

if file:

    # ---- Load image ----
    if file.name.endswith(".mat"):
        with h5py.File(file, 'r') as f:
            img = np.array(f['cjdata']['image']).T
    else:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 0)

    img = cv2.resize(img, (256, 256))

    # ---- FIX IMAGE ERROR ----
    img = np.nan_to_num(img)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)

    st.image(img, caption="Input MRI Image", use_container_width=True)

    # ---- FEATURES ----
    features = extract_features(img)
    X = np.array(features).reshape(1, -1)

    # ---- PREPROCESS ----
    X = imputer.transform(X)
    X = scaler.transform(X)

    # ---- PREDICT ----
    if st.button("Predict"):
        pred = model.predict(X)
        prob = model.predict_proba(X)

        st.success(f"🧠 Prediction: {pred[0]}")
        st.info(f"Confidence: {np.max(prob)*100:.2f}%")
