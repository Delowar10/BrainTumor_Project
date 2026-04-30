import os
import cv2
import h5py
import numpy as np
import warnings

from scipy.stats import skew, kurtosis
from statsmodels.stats.stattools import durbin_watson

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# =========================
# DATA PATH
# =========================
DATA_DIR = "/home/santo/Downloads/paper thesis/1512427"

folders = sorted([
    "brainTumorDataPublic_1-766",
    "brainTumorDataPublic_767-1532",
    "brainTumorDataPublic_1533-2298",
    "brainTumorDataPublic_2299-3064"
])

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(img):
    p = img.flatten().astype(np.float64)
    p = np.nan_to_num(p)

    return [
        np.mean(p), np.std(p), np.var(p),
        np.min(p), np.max(p), np.median(p),
        skew(p), kurtosis(p),
        durbin_watson(p)
    ]

# =========================
# LOAD DATA
# =========================
X, y = [], []

print("Loading dataset...")

for folder in folders:
    folder_path = os.path.join(DATA_DIR, folder)

    for file in os.listdir(folder_path):
        if file.endswith(".mat"):
            with h5py.File(os.path.join(folder_path, file), 'r') as f:
                img = np.array(f['cjdata']['image']).T
                label = int(np.array(f['cjdata']['label'])[0][0]) - 1

            img = cv2.resize(img, (256, 256))
            features = extract_features(img)

            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset Shape:", X.shape)

# =========================
# PREPROCESSING
# =========================
imputer = KNNImputer(n_neighbors=5)
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# MODELS (CORRECTED)
# =========================
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=200),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=200),
    "LogisticRegression": LogisticRegression(max_iter=1000),

    "CatBoost": CatBoostClassifier(
        iterations=200,
        verbose=0
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )
}

# =========================
# CROSS VALIDATION
# =========================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_score = 0
best_model = None
best_name = ""

print("\n===== MODEL COMPARISON =====\n")

for name, model in models.items():

    acc_list = []

    for train_idx, test_idx in skf.split(X, y):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc_list.append(accuracy_score(y_test, preds))

    mean_acc = np.mean(acc_list)

    print(f"{name}: {mean_acc*100:.2f}%")

    if mean_acc > best_score:
        best_score = mean_acc
        best_model = model
        best_name = name

# =========================
# FINAL MODEL SAVE
# =========================
import joblib

best_model.fit(X, y)

joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")

print("\n===== FINAL RESULT =====")
print(f"Best Model: {best_name}")
print(f"Accuracy: {best_score*100:.2f}%")
print("Model saved successfully!")
