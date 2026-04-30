import streamlit as st
import numpy as np
import joblib
import cv2
import h5py

from scipy.stats import skew, kurtosis
from statsmodels.stats.stattools import durbin_watson

# =====================
# LOAD MODEL
# =====================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

st.title("🧠 Brain Tumor Classification System")

# =====================
# FEATURE EXTRACTION
# =====================
def extract_features(img):
    p = img.flatten().astype(np.float64)
    p = np.nan_to_num(p)

    return [
        np.mean(p), np.std(p), np.var(p),
        np.min(p), np.max(p), np.median(p),
        skew(p), kurtosis(p),
        durbin_watson(p)
    ]

# =====================
# UPLOAD FILE
# =====================
file = st.file_uploader("Upload MRI (.mat or image)")

if file:

    if file.name.endswith(".mat"):
        with h5py.File(file, 'r') as f:
            img = np.array(f['cjdata']['image']).T
    else:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 0)

    img = cv2.resize(img, (256, 256))
    st.image(img, caption="Input Image")

    features = extract_features(img)
    X = np.array(features).reshape(1, -1)

    # preprocessing
    X = imputer.transform(X)
    X = scaler.transform(X)

    if st.button("Predict"):
        pred = model.predict(X)
        st.success(f"Prediction: Class {pred[0]}")