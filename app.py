import streamlit as st
import numpy as np
import joblib
import os
import h5py
import cv2

from utils import extract_features

# =====================
# LOAD MODEL (SAFE PATH)
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
imputer = joblib.load(os.path.join(BASE_DIR, "imputer.pkl"))

# =====================
# UI
# =====================
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

st.title("🧠 Brain Tumor Classification System")
st.write("Upload an MRI image (.jpg, .png or .mat)")

# =====================
# FILE UPLOAD
# =====================
file = st.file_uploader("Upload MRI file")

if file:
    try:
        if file.name.endswith(".mat"):
            with h5py.File(file, 'r') as f:
                keys = list(f.keys())
                st.write("MAT Keys:", keys)
                img = np.array(f[keys[0]]).T
        else:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 0)

        if img is None:
            st.error("❌ Image not loaded properly")
        else:
            img = cv2.resize(img, (256, 256))
            st.image(img, caption="Uploaded Image", use_column_width=True)

            features = extract_features(img)
            X = np.array(features).reshape(1, -1)

            # preprocess
            X = imputer.transform(X)
            X = scaler.transform(X)

            if st.button("🔍 Predict"):
                pred = model.predict(X)[0]
                prob = model.predict_proba(X)[0]

                st.success(f"Prediction: Class {pred}")
                st.info(f"Confidence: {np.max(prob)*100:.2f}%")

    except Exception as e:
        st.error(f"Error: {e}")
