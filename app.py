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
MODEL_FILE_ID = "1Cu8QVXtO-YTVMBidQuIKibEINuO8wcU_"   # ✔ তোমারটা দেওয়া আছে
SCALER_FILE_ID = "PASTE_YOUR_SCALER_ID"
IMPUTER_FILE_ID = "PASTE_YOUR_IMPUTER_ID"

# =========================
# DOWNLOAD FUNCTION
# =========================
def download_file(file_id, output_name):
    if not os.path.exists(output_name):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        try:
            gdown.download(url, output_name, quiet=False)
        except:
            st.error(f"❌ Failed to download {output_name}. Check Drive sharing!")

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
# UI DESIGN
# =========================
st.set_page_config(page_title="Brain Tumor AI", layout="centered")

st.title("🧠 Brain Tumor Classification System")
st.write("Upload MRI image (.mat or image file)")

# =========================
# FILE UPLOAD
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

    st.image(img, caption="Input MRI Image", use_container_width=True)

    # ---- Feature extraction ----
    features = extract_features(img)
    X = np.array(features).reshape(1, -1)

    # ---- Preprocessing ----
    X = imputer.transform(X)
    X = scaler.transform(X)

    # ---- Prediction ----
    if st.button("Predict Tumor Type"):
        pred = model.predict(X)
        prob = model.predict_proba(X)

        st.success(f"🧠 Prediction Class: {pred[0]}")
        st.info(f"Confidence: {np.max(prob)*100:.2f}%")
