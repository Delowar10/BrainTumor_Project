import numpy as np
from scipy.stats import skew, kurtosis
from statsmodels.stats.stattools import durbin_watson

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
