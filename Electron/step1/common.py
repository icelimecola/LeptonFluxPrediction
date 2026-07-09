#!/bin/python

from pathlib import Path
import re

import numpy as np
import pandas as pd


STEP1_DIR = Path(__file__).resolve().parent
ELECTRON_DIR = STEP1_DIR.parent
PAD_DAYS = 365
FLUX_START = pd.Timestamp("2011-06-11")
SUN_START = pd.Timestamp("2010-05-20")
SUN_OFFSET = (FLUX_START - pd.Timedelta(days=PAD_DAYS) - SUN_START).days


def ensure_dirs():
    (STEP1_DIR / "Data" / "flux").mkdir(parents=True, exist_ok=True)
    (STEP1_DIR / "Data" / "model").mkdir(parents=True, exist_ok=True)
    (STEP1_DIR / "Data" / "lstmdraw").mkdir(parents=True, exist_ok=True)
    (STEP1_DIR / "Figure" / "flux").mkdir(parents=True, exist_ok=True)
    (STEP1_DIR / "Figure" / "lstmtrain").mkdir(parents=True, exist_ok=True)
    (STEP1_DIR / "Figure" / "lstmdraw").mkdir(parents=True, exist_ok=True)


def imputation_path(name):
    return STEP1_DIR / "Data" / "flux" / name


def draw_data_path(name):
    return STEP1_DIR / "Data" / "lstmdraw" / name


def model_dir():
    return STEP1_DIR / "Data" / "model"


def figure_dir():
    return draw_figure_dir()


def flux_figure_dir():
    return STEP1_DIR / "Figure" / "flux"


def train_figure_dir():
    return STEP1_DIR / "Figure" / "lstmtrain"


def draw_figure_dir():
    return STEP1_DIR / "Figure" / "lstmdraw"


def load_sun_daily():
    return np.load(ELECTRON_DIR.parent / "sun_processed" / "latest" / "sun5_daily_all_latest.npy")


def valid_observation(flux, mask=None):
    valid = np.isfinite(flux) & (flux > 0)
    if mask is not None:
        valid &= mask.astype(bool)
    return valid


def make_sequence(*arrays, look_back):
    seqs = [[] for _ in arrays]
    for i in range(look_back, len(arrays[0])):
        for out, arr in zip(seqs, arrays):
            out.append(arr[i - look_back:i] if arr.ndim == 2 else arr[i - look_back:i, ...])
    return [np.array(out) for out in seqs]


def make_target_sequence(y, mask, look_back):
    x_seq = []
    y_seq = []
    m_seq = []
    for i in range(look_back, len(y)):
        x_seq.append(i)
        y_seq.append(y[i, :])
        m_seq.append(mask[i, :])
    return np.array(x_seq), np.array(y_seq), np.array(m_seq)


def nanminmax_fit(data, mask=None, min_span=1e-12):
    data = np.asarray(data, dtype=float)
    if mask is None:
        mask = np.isfinite(data)
    else:
        mask = mask.astype(bool) & np.isfinite(data)
    n_cols = data.shape[1]
    data_min = np.zeros(n_cols, dtype=float)
    data_max = np.ones(n_cols, dtype=float)
    for j in range(n_cols):
        valid = mask[:, j]
        if np.any(valid):
            data_min[j] = np.nanmin(data[valid, j])
            data_max[j] = np.nanmax(data[valid, j])
        span = data_max[j] - data_min[j]
        if not np.isfinite(span) or abs(span) < min_span:
            data_max[j] = data_min[j] + 1.0
    return data_min, data_max


def minmax_transform(data, data_min, data_max, fill_value=0.0):
    data = np.asarray(data, dtype=float)
    scaled = (data - data_min) / (data_max - data_min)
    return np.where(np.isfinite(scaled), scaled, fill_value)


def minmax_inverse(scaled, data_min, data_max):
    return np.asarray(scaled, dtype=float) * (data_max - data_min) + data_min


def interpolate_nan_by_bin(data):
    idx = pd.RangeIndex(data.shape[0])
    filled = np.zeros_like(data, dtype=float)
    for j in range(data.shape[1]):
        series = pd.Series(data[:, j], index=idx)
        series = series.interpolate(limit_direction="both")
        series = series.ffill().bfill()
        if series.isna().any():
            series = series.fillna(0.0)
        filled[:, j] = series.to_numpy()
    return filled


def relative_rmse_by_bin(pred, true, mask, fallback=0.3, min_points=10):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    mask = np.asarray(mask).astype(bool)
    n_bins = true.shape[1]
    out = np.full(n_bins, np.nan, dtype=float)
    for j in range(n_bins):
        valid = valid_observation(true[:, j], mask[:, j]) & np.isfinite(pred[:, j])
        if np.sum(valid) >= min_points:
            rel = (pred[valid, j] - true[valid, j]) / true[valid, j]
            out[j] = np.sqrt(np.mean(rel ** 2))
    if np.any(np.isfinite(out)):
        global_fallback = np.nanmedian(out)
    else:
        global_fallback = fallback
    out = np.where(np.isfinite(out), out, global_fallback)
    return out


def find_best_model(prefix="sunImputer_", prefer_best_file=True):
    best_file = model_dir() / "best_model.txt"
    if prefer_best_file and best_file.exists():
        candidate = best_file.read_text(encoding="utf-8").strip()
        path = model_dir() / candidate
        if path.exists():
            return path
    pattern = re.compile(rf"^{re.escape(prefix)}.*_(?P<epoch>\d+)-(?P<val>[-+0-9.eE]+)\.keras$")
    rows = []
    for path in model_dir().glob(f"{prefix}*.keras"):
        match = pattern.match(path.name)
        if match:
            rows.append((float(match.group("val")), int(match.group("epoch")), path.name, path))
    if not rows:
        raise FileNotFoundError(f"No {prefix}*.keras model found in {model_dir()}")
    rows.sort()
    return rows[0][3]
