#!/bin/python

import argparse

import numpy as np
import pandas as pd

from common import draw_data_path, figure_dir, imputation_path, interpolate_nan_by_bin, relative_rmse_by_bin, valid_observation


def parse_args():
    parser = argparse.ArgumentParser(description="Compare linear interpolation and sun-only predictions on artificial gaps.")
    parser.add_argument("--lengths", type=int, nargs="+", default=[30, 60, 120])
    parser.add_argument("--windows-per-length", type=int, default=5)
    parser.add_argument("--min-observed-frac", type=float, default=0.95)
    return parser.parse_args()


def choose_windows(mask_any, length, n_windows, min_observed_frac):
    candidates = []
    max_start = len(mask_any) - length
    if max_start <= 0:
        return candidates
    starts = np.linspace(0, max_start, max_start + 1, dtype=int)
    valid_starts = []
    for s in starts:
        frac = np.mean(mask_any[s:s + length])
        if frac >= min_observed_frac:
            valid_starts.append(s)
    if not valid_starts:
        return candidates
    pick_idx = np.linspace(0, len(valid_starts) - 1, min(n_windows, len(valid_starts)), dtype=int)
    for idx in pick_idx:
        s = valid_starts[idx]
        candidates.append((s, s + length))
    return candidates


def safe_mean_relative_rmse(pred, true, mask):
    per_bin = relative_rmse_by_bin(pred, true, mask, min_points=3)
    return float(np.nanmedian(per_bin))


def main():
    args = parse_args()
    flux_obs = np.load(imputation_path("electron_flux_observed_nan.npy"))
    mask_obs = np.load(imputation_path("electron_observed_mask.npy")).astype(bool)
    sun_pred = np.load(imputation_path("electron_flux_sun_pred_allbin.npy"))

    mask_any = np.mean(mask_obs, axis=1) >= args.min_observed_frac
    rows = []
    for length in args.lengths:
        for start, end in choose_windows(mask_any, length, args.windows_per_length, args.min_observed_frac):
            artificial_mask = mask_obs.copy()
            artificial_mask[start:end, :] = False
            flux_gap = flux_obs.copy()
            flux_gap[start:end, :] = np.nan
            linear = interpolate_nan_by_bin(flux_gap)

            eval_mask = mask_obs[start:end, :] & valid_observation(flux_obs[start:end, :])
            true = flux_obs[start:end, :]
            lin_score = safe_mean_relative_rmse(linear[start:end, :], true, eval_mask)
            sun_score = safe_mean_relative_rmse(sun_pred[start:end, :], true, eval_mask)
            rows.append({
                "gap_length": length,
                "start_index": start,
                "end_index": end,
                "linear_median_relative_rmse": lin_score,
                "sun_lstm_median_relative_rmse": sun_score,
                "valid_points": int(np.sum(eval_mask)),
            })

    df = pd.DataFrame(rows)
    out_csv = draw_data_path("masked_gap_validation_summary.csv")
    df.to_csv(out_csv, index=False)

    if not df.empty:
        import matplotlib.pyplot as plt

        labels = [str(v) for v in df["gap_length"]]
        x = np.arange(len(df))
        plt.figure(figsize=(max(8, len(df) * 0.45), 4))
        plt.plot(x, df["linear_median_relative_rmse"], "o-", label="linear")
        plt.plot(x, df["sun_lstm_median_relative_rmse"], "o-", label="sun-only LSTM")
        plt.xticks(x, labels, rotation=45)
        plt.ylabel("median relative RMSE")
        plt.xlabel("artificial gap length")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir() / "masked_gap_validation_summary.pdf")
        plt.close()

    print("Saved validation summary:", out_csv)
    print(df)


if __name__ == "__main__":
    main()
