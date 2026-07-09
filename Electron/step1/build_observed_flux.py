#!/bin/python

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import uproot

from common import ELECTRON_DIR, ensure_dirs, imputation_path


def timestamp_to_date(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def main():
    ensure_dirs()
    root_path = ELECTRON_DIR.parent / "rawdata" / "flux" / "eleflux.root"
    f = uproot.open(root_path)
    h2 = f["hflux_2d;1"]

    flux_2d = h2.values().astype(float)
    err_2d = h2.errors().astype(float)

    x_low = h2.axis(0).member("fXmin")
    x_high = h2.axis(0).member("fXmax")
    x_nbins = h2.axis(0).member("fNbins")
    bin_width = (x_high - x_low) / x_nbins

    y_nbins = h2.axis(1).member("fNbins")
    y_edges = np.array(h2.axis(1).member("fXbins"))
    keep_bins = np.arange(1, y_nbins)
    flux_2d = flux_2d[:, keep_bins]
    err_2d = err_2d[:, keep_bins]
    y_edges_kept = y_edges[keep_bins[0]:]

    start_date = timestamp_to_date(x_low + bin_width / 2)
    end_date = timestamp_to_date(x_high - bin_width / 2)
    idx = pd.date_range(start=start_date, end=end_date, freq="D")[: flux_2d.shape[0]]

    has_data = np.sum(flux_2d, axis=1) > 0
    valid_rows = np.where(has_data)[0]
    if len(valid_rows) == 0:
        raise ValueError("No nonzero electron flux rows found in ROOT file.")

    first_valid = valid_rows[0]
    last_valid = valid_rows[-1]
    flux_2d = flux_2d[first_valid:last_valid + 1, :]
    err_2d = err_2d[first_valid:last_valid + 1, :]
    idx = idx[first_valid:last_valid + 1]

    observed_mask = np.isfinite(flux_2d) & (flux_2d > 0) & np.isfinite(err_2d)
    flux_nan = flux_2d.copy()
    err_nan = err_2d.copy()
    flux_nan[~observed_mask] = np.nan
    err_nan[~observed_mask] = np.nan

    np.save(imputation_path("electron_flux_observed_nan.npy"), flux_nan)
    np.save(imputation_path("electron_err_observed_nan.npy"), err_nan)
    np.save(imputation_path("electron_observed_mask.npy"), observed_mask.astype(np.uint8))
    np.save(imputation_path("electron_energy_edges.npy"), y_edges_kept)

    meta = pd.DataFrame({"date": idx.strftime("%Y-%m-%d")})
    meta.to_csv(imputation_path("electron_observed_dates.csv"), index=False)

    print("Saved observed flux with NaNs:")
    print("  flux:", flux_nan.shape, imputation_path("electron_flux_observed_nan.npy"))
    print("  err: ", err_nan.shape, imputation_path("electron_err_observed_nan.npy"))
    print("  mask:", observed_mask.shape, imputation_path("electron_observed_mask.npy"))
    print("  dates:", idx[0].date(), "to", idx[-1].date())
    print("  missing entries:", int(np.sum(~observed_mask)), "/", observed_mask.size)


if __name__ == "__main__":
    main()
