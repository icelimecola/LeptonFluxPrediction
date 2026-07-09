#!/bin/python

import argparse

import numpy as np

from common import (
    PAD_DAYS,
    SUN_OFFSET,
    ensure_dirs,
    find_best_model,
    imputation_path,
    load_sun_daily,
    minmax_inverse,
    minmax_transform,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict missing electron flux with the sun-only imputer.")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--flux", type=str, default=str(imputation_path("electron_flux_observed_nan.npy")))
    parser.add_argument("--mask", type=str, default=str(imputation_path("electron_observed_mask.npy")))
    return parser.parse_args()


def make_sequence(X, look_back):
    return np.array([X[i - look_back:i, :] for i in range(look_back, len(X))])


def main():
    ensure_dirs()
    args = parse_args()

    import tensorflow as tf
    from tensorflow.keras.models import load_model

    def masked_huber(y_packed, y_pred, delta=1.0):
        bins = y_pred.shape[-1]
        y_true = y_packed[:, :bins]
        mask = y_packed[:, bins:]
        abs_error = tf.abs(y_true - y_pred)
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic
        loss = 0.5 * tf.square(quadratic) + delta * linear
        denom = tf.maximum(tf.reduce_sum(mask), 1.0)
        return tf.reduce_sum(loss * mask) / denom

    flux_obs = np.load(args.flux)
    mask_obs = np.load(args.mask).astype(bool)
    scalers = np.load(imputation_path("sun_imputer_scalers.npz"))
    look_back = int(scalers["look_back"])
    x_min, x_max = scalers["x_min"], scalers["x_max"]
    y_min, y_max = scalers["y_min"], scalers["y_max"]

    model_path = args.model if args.model else str(find_best_model("sunImputer_"))
    model = load_model(model_path, custom_objects={"masked_huber": masked_huber})

    bins = flux_obs.shape[1]
    flux_padded = np.concatenate([np.full((PAD_DAYS, bins), np.nan), flux_obs], axis=0)
    sun_daily = load_sun_daily()
    X_all = sun_daily[SUN_OFFSET:SUN_OFFSET + len(flux_padded), 0:5]
    X_all = minmax_transform(X_all, x_min, x_max)
    X_seq = make_sequence(X_all, look_back)

    pred_scaled = model.predict(X_seq, verbose=1)
    pred_all_targets = minmax_inverse(pred_scaled, y_min, y_max)

    target_indices = np.arange(look_back, len(flux_padded))
    actual_rows = target_indices - PAD_DAYS
    keep = (actual_rows >= 0) & (actual_rows < len(flux_obs))
    pred_flux = pred_all_targets[keep]
    if pred_flux.shape != flux_obs.shape:
        raise ValueError(f"Prediction shape {pred_flux.shape} does not match observed flux shape {flux_obs.shape}")

    flux_imputed = flux_obs.copy()
    flux_imputed[~mask_obs] = pred_flux[~mask_obs]

    np.save(imputation_path("electron_flux_sun_pred_allbin.npy"), pred_flux)
    np.save(imputation_path("electron_flux_sun_imputed.npy"), flux_imputed)

    print("Loaded model:", model_path)
    print("Saved all sun-only predictions:", imputation_path("electron_flux_sun_pred_allbin.npy"))
    print("Saved imputed flux:", imputation_path("electron_flux_sun_imputed.npy"))
    print("Remaining NaN:", int(np.sum(~np.isfinite(flux_imputed))))


if __name__ == "__main__":
    main()

