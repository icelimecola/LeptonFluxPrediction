#!/usr/bin/env python3
"""Train the n2p (neutron -> proton) residual network with PyTorch.

Inputs
------
X : the SAITS-imputed NM daily count-rate matrix
    (default ``NM/data/nmdb_imputed/nm_daily_imputed_counts.npy``, (T, 18))
y : the AMS daily proton flux, dropna version (no interpolation), only days
    where all rigidity bins are present
    (default ``PROTON/data/proton_allbin_dropna.npy``, (2717, 30))

The proton array's own dates are used to pick the matching neutron rows, so
the two sides are aligned by date (not by row index).

Model (same architecture as 小导's Keras n2p, ported to PyTorch)
----------------------------------------------------------------
Adaptive-N Conv1D residual network:

    Input (N, 1)
    -> Conv1d(64, k1)+BN+ReLU+Dropout
    -> Conv1d(64, k2)+BN+ReLU+Dropout
    -> Conv1d(64, k3)+BN+ReLU+Dropout
    -> 3 x residual block (3 x Conv1d(64; k1,k2,k3)+BN+ReLU+Dropout, Add shortcut)
    -> GlobalAveragePooling
    -> Linear(n_bins)  (L2 applied to this layer only)

with k = [min(7, N), min(5, N), min(3, N)].

Training mirrors 小导: log10 target, MinMax scaling (fit on train only),
(``--no-log`` skips the log10 and MinMax-scales the raw flux instead, for the
log-vs-linear target comparison; the default is unchanged),
random 80/10/10 split controlled by --data-seed, Adamax, MSE loss,
early stopping on val_loss + keep the best-val weights.

Keras-equivalence details (2026-09-27):
  * L2: Keras ``L2(l2)`` on the Dense kernel adds ``l2 * sum(W**2)`` to the
    loss (gradient ``2*l2*W``), leaves the bias alone, and is included in the
    reported train/val/test loss. Reproduced here as an explicit penalty on
    ``head.weight`` only, added to every reported loss, so early stopping
    compares the same quantity as Keras' ``val_loss``.
  * BatchNorm: Keras defaults momentum=0.99, epsilon=1e-3; PyTorch's momentum
    is the complement, so ``BatchNorm1d(momentum=0.01, eps=1e-3)``.
  * Dates: the neutron rows are dated from the ``date`` column of the CSV that
    sits next to the .npy (written by nmdb_impute_saits.py). ``--neutron-start``
    is only a fallback when no CSV is present; if both exist they must agree.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import sys
from pathlib import Path

import numpy as np


# --- default station names (Rc ascending), same order as the NM pipeline ---
try:  # prefer the shared metadata when running from the repo
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nmdb_download import DEFAULT_STATIONS as _DEFAULT_STATIONS  # type: ignore
    from station_metadata import (  # type: ignore
        stations_by_cutoff_rigidity as _by_rc,
    )

    DEFAULT_STATION_ORDER = _by_rc(list(_DEFAULT_STATIONS))
except Exception:  # pragma: no cover - fallback keeps the script self-contained
    DEFAULT_STATION_ORDER = [
        "TERA", "SOPB", "SOPO", "FSMT", "INVK", "NAIN", "PWNK", "THUL",
        "APTY", "OULU", "YKTK", "NEWK", "LMKS", "JUNG", "JUNG1", "AATB",
        "MXCO", "PSNM",
    ]


# --------------------------------------------------------------------------
# data loading / alignment (pure numpy; importable without torch)
# --------------------------------------------------------------------------
def read_neutron_csv_dates(npy_path: Path) -> list[dt.date] | None:
    """Dates from the ``date`` column of the CSV next to the .npy, if any."""
    csv_path = npy_path.with_suffix(".csv")
    if not csv_path.is_file():
        return None
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header or header[0].strip().lower() != "date":
            return None
        return [dt.date.fromisoformat(row[0].strip()) for row in reader if row]


def read_neutron(path: Path, start_date: dt.date | None):
    """Load the neutron matrix and date its rows.

    The CSV date column is authoritative. ``start_date`` (--neutron-start) is
    used only when there is no CSV; if both are available they must agree, so
    a stale start date can never silently shift the whole series.
    """
    data = np.load(path)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise SystemExit(f"neutron matrix must be 2-D, got {data.shape}")
    csv_dates = read_neutron_csv_dates(path)
    if csv_dates is not None:
        if len(csv_dates) != data.shape[0]:
            raise SystemExit(
                f"CSV next to {path.name} has {len(csv_dates)} dated rows but "
                f"the matrix has {data.shape[0]}"
            )
        steps = {(b - a).days for a, b in zip(csv_dates, csv_dates[1:])}
        if steps and steps != {1}:
            raise SystemExit("neutron CSV dates are not a consecutive daily grid")
        if start_date is not None and start_date != csv_dates[0]:
            raise SystemExit(
                f"--neutron-start {start_date} disagrees with the CSV, which "
                f"starts at {csv_dates[0]}; drop --neutron-start or fix it"
            )
        print(f"neutron dates from CSV: {csv_dates[0]} .. {csv_dates[-1]}")
        return data, csv_dates
    if start_date is None:
        raise SystemExit(
            f"no dated CSV next to {path}; pass --neutron-start YYYY-MM-DD"
        )
    dates = [start_date + dt.timedelta(days=i) for i in range(data.shape[0])]
    print(f"neutron dates from --neutron-start: {dates[0]} .. {dates[-1]}")
    return data, dates


def read_neutron_station_names(npy_path: Path, n_cols: int) -> list[str]:
    """Station names for each neutron column, read from the sibling CSV header.

    Falls back to the Rc-ordered default list truncated to n_cols when no CSV
    is present next to the .npy.
    """
    csv_path = npy_path.with_suffix(".csv")
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            header = next(csv.reader(stream), None)
        if header and len(header) - 1 == n_cols:
            return [name.strip() for name in header[1:]]
    return list(DEFAULT_STATION_ORDER[:n_cols])


def read_proton(path: Path, meta_path: Path):
    flux = np.asarray(np.load(path), dtype=np.float64)
    if flux.ndim != 2:
        raise SystemExit(f"proton matrix must be 2-D, got {flux.shape}")
    if not meta_path.is_file():
        raise SystemExit(f"proton meta not found: {meta_path}")
    with np.load(meta_path) as meta:
        dates = [dt.date.fromisoformat(str(text)) for text in meta["dates"]]
    if len(dates) != flux.shape[0]:
        raise SystemExit(
            f"proton dates ({len(dates)}) do not match rows ({flux.shape[0]})"
        )
    return flux, dates


def read_bins(meta_path: Path):
    with np.load(meta_path) as meta:
        return [
            (float(a), float(b))
            for a, b in zip(meta["rigidity_min"], meta["rigidity_max"])
        ]


def align_rows(neutron_dates, proton_dates) -> np.ndarray:
    """Return neutron row indices matching the proton dates, in order."""
    lookup = {day: idx for idx, day in enumerate(neutron_dates)}
    missing = [day for day in proton_dates if day not in lookup]
    if missing:
        raise SystemExit(
            f"{len(missing)} proton dates are outside the neutron date range, "
            f"first: {missing[0]}"
        )
    return np.asarray([lookup[day] for day in proton_dates], dtype=int)


def select_station_indices(station_names, requested) -> list[int]:
    """Validate requested station names; keep matrix (Rc) order."""
    if requested is None:
        return list(range(len(station_names)))
    if isinstance(requested, str):
        requested = [part for part in requested.split(",")]
    wanted = [name.strip().upper() for name in requested if name.strip()]
    unknown = [name for name in wanted if name not in station_names]
    if unknown:
        raise SystemExit(
            "unknown station(s): " + ", ".join(unknown)
            + "\navailable: " + ", ".join(station_names)
        )
    wanted_set = set(wanted)
    return [i for i, name in enumerate(station_names) if name in wanted_set]


# Wording of the plots. nmdb_n2p.py is proton / rigidity (GV); other scripts that
# reuse the plotting helpers (e.g. nmdb_p2pos.py, positron / energy in GeV)
# update this dict before plotting. Defaults leave the n2p figures unchanged.
PLOT_LABELS = {"particle": "proton", "unit": "GV", "axis": "rigidity",
               "err": "stat+syst"}


def _flux_label() -> str:
    unit = PLOT_LABELS["unit"]
    return "flux (m$^{-2}$sr$^{-1}$s$^{-1}$" + unit + "$^{-1}$)"


def adaptive_kernels(n_stations: int) -> list[int]:
    """Kernel sizes clipped to the number of stations (mirrors 7/5/3)."""
    kernels = sorted({min(7, n_stations), min(5, n_stations), min(3, n_stations)})
    return list(reversed(kernels))


# --------------------------------------------------------------------------
# model (PyTorch)
# --------------------------------------------------------------------------
def build_model(n_stations: int, n_bins: int, dropout: float, init: str = "keras"):
    import torch
    import torch.nn as nn

    kernels = adaptive_kernels(n_stations)

    def conv_stack():
        layers = []
        for k in kernels:
            layers += [
                nn.Conv1d(64, 64, k, padding="same"),
                nn.BatchNorm1d(64, momentum=0.01, eps=1e-3),  # Keras defaults
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        return nn.Sequential(*layers)

    class ResidualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = conv_stack()
            self.out_act = nn.ReLU()
            self.out_drop = nn.Dropout(dropout)

        def forward(self, x):
            return self.out_drop(self.out_act(self.body(x) + x))

    class N2PResidualNet(nn.Module):
        def __init__(self):
            super().__init__()
            first = []
            for k in kernels:
                first += [
                    nn.Conv1d(1 if not first else 64, 64, k, padding="same"),
                    nn.BatchNorm1d(64, momentum=0.01, eps=1e-3),  # Keras defaults
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            self.input_block = nn.Sequential(*first)
            self.res_blocks = nn.Sequential(*[ResidualBlock() for _ in range(3)])
            self.head = nn.Linear(64, n_bins)

        def forward(self, x):                      # x: (B, N, 1)
            x = x.transpose(1, 2)                  # -> (B, 1, N)
            x = self.input_block(x)
            x = self.res_blocks(x)
            x = x.mean(dim=2)                      # global average pooling
            return self.head(x)

    model = N2PResidualNet()
    if init == "keras":
        # Keras defaults: glorot_uniform kernels, zero biases (Conv1D and Dense);
        # BatchNorm gamma=1, beta=0 is already the same in both frameworks.
        # PyTorch's own default (Kaiming-uniform) is kept with init="torch".
        for module in model.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    elif init != "torch":
        raise ValueError(f"unknown init {init!r} (use 'keras' or 'torch')")
    return model, kernels


def inverse_log_minmax(values: np.ndarray, scaler) -> np.ndarray:
    """scaled -> log10(flux) -> flux."""
    return np.power(10.0, scaler.inverse_transform(values))


def inverse_target(values: np.ndarray, scaler, y_is_log10: bool = True) -> np.ndarray:
    """scaled -> flux; log10 target (default) or raw-flux target (--no-log)."""
    if y_is_log10:
        return inverse_log_minmax(values, scaler)
    return scaler.inverse_transform(values)


def write_scalers_csv(path: Path, stations, x_min, x_max, y_min, y_max,
                      bins, y_is_log10: bool) -> None:
    """Human-readable copy of scalers.npz (same numbers), one row per station / bin.

    range = data_max - data_min, in the units MinMax worked in (raw counts for
    x; log10(flux) or raw flux for y). max_over_min is always in physical units.
    """
    transform = "log10" if y_is_log10 else "linear"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "index", "name", "rigidity_min_GV", "rigidity_max_GV",
                         "data_min", "data_max", "range", "max_over_min", "transform"])
        for i, (name, lo, hi) in enumerate(zip(stations, x_min, x_max), start=1):
            writer.writerow(["x_neutron", i, name, "", "", f"{lo:.10g}", f"{hi:.10g}",
                             f"{hi - lo:.10g}", f"{hi / lo:.10g}", "none"])
        for i, (lo, hi) in enumerate(zip(y_min, y_max), start=1):
            rmin, rmax = bins[i - 1] if bins is not None else ("", "")
            ratio = 10.0 ** (hi - lo) if y_is_log10 else hi / lo
            writer.writerow(["y_proton", i, f"bin{i:02d}", rmin, rmax, f"{lo:.10g}",
                             f"{hi:.10g}", f"{hi - lo:.10g}", f"{ratio:.10g}", transform])


def predict_flux(model, X, scaler_y, device, y_is_log10: bool = True):
    import torch

    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        pred_scaled = model(tensor).cpu().numpy()
    return inverse_target(pred_scaled, scaler_y, y_is_log10)


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def make_plots(out_dir, train_losses, val_losses, test_losses, y_true, y_pred, bins):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1) loss curve (train / validation / test)
    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    epochs = range(1, len(train_losses) + 1)
    axis.plot(epochs, train_losses, label="training")
    axis.plot(epochs, val_losses, label="validation")
    if test_losses:
        axis.plot(range(1, len(test_losses) + 1), test_losses, label="test")
    axis.set_yscale("log")
    axis.set_title("model loss")
    axis.set_ylabel("loss")
    axis.set_xlabel("epoch")
    axis.legend(loc="upper right", fontsize=9, frameon=False)
    figure.tight_layout()
    path = out_dir / "model_loss.pdf"
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    paths.append(str(path))

    # 2) predicted vs observed, all bins (log-log)
    figure, axis = plt.subplots(figsize=(6.0, 6.0))
    axis.plot(y_true.ravel(), y_pred.ravel(), ".", markersize=2,
              color="#1a73e8", markeredgewidth=0)
    lo = float(np.nanmin([y_true, y_pred]))
    hi = float(np.nanmax([y_true, y_pred]))
    axis.plot([lo, hi], [lo, hi], color="0.3", linewidth=1.0, linestyle=":")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("AMS Data flux")
    axis.set_ylabel("predicted flux")
    axis.set_title("test: predicted vs AMS Data")
    figure.tight_layout()
    path = out_dir / "scatter_all_bins.pdf"
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    paths.append(str(path))

    # 3) per-bin relative error
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(y_pred - y_true) / y_true
    rel_mean = np.nanmean(rel, axis=0)
    figure, axis = plt.subplots(figsize=(9.0, 4.0))
    axis.bar(range(1, len(rel_mean) + 1), rel_mean * 100.0, color="#26734d")
    axis.set_xlabel(f"{PLOT_LABELS['axis']} bin index")
    axis.set_ylabel("mean |relative error| (%)")
    axis.set_title("test: per-bin relative error")
    figure.tight_layout()
    path = out_dir / "per_bin_relative_error.pdf"
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    paths.append(str(path))

    # 4) per-bin scatter, grouped 6 bins per figure (3 x 2), like the imputed groups
    n_bins = y_true.shape[1]
    for group_start in range(0, n_bins, 6):
        group = list(range(group_start, min(group_start + 6, n_bins)))
        group_number = group_start // 6 + 1
        figure, axes = plt.subplots(3, 2, figsize=(13.0, 10.0), squeeze=False)
        for index, j in enumerate(group):
            axis = axes[index // 2, index % 2]
            axis.plot(y_true[:, j], y_pred[:, j], ".", markersize=2.6,
                      color="#5f6368", markeredgewidth=0)
            lo = float(min(y_true[:, j].min(), y_pred[:, j].min()))
            hi = float(max(y_true[:, j].max(), y_pred[:, j].max()))
            axis.plot([lo, hi], [lo, hi], color="0.3", linewidth=0.8, linestyle=":")
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel("AMS Data", fontsize=9)
            axis.set_ylabel("predicted", fontsize=9)
            axis.set_title(f"bin {j + 1}: {bins[j][0]:g}-{bins[j][1]:g} {PLOT_LABELS['unit']}",
                           fontsize=11, fontweight="bold")
            axis.tick_params(direction="in", top=True, right=True, labelsize=8)
        for index in range(len(group), 6):
            axes[index // 2, index % 2].set_visible(False)
        figure.suptitle(
            "test: predicted vs AMS Data (log-log)",
            fontsize=14, fontweight="bold",
        )
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        path = out_dir / f"combined_scatter_group_{group_number}.pdf"
        figure.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        paths.append(str(path))
    return paths


# --------------------------------------------------------------------------
# time-series plots: observed vs predicted, per bin (individual + grouped)
# --------------------------------------------------------------------------
def make_timeseries_plots(
    indiv_dir, combined_dir, date_splits, true_by_split, pred_by_split, bins,
    group_size=6,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    records = []
    for label in ("train", "val", "test"):
        dates = date_splits[label]
        truth = true_by_split[label]
        pred = pred_by_split[label]
        for k in range(len(dates)):
            records.append((dates[k], truth[k], pred[k], label))
    records.sort(key=lambda row: row[0])
    dates = [row[0] for row in records]
    obs = np.asarray([row[1] for row in records])
    prd = np.asarray([row[2] for row in records])
    split_of = np.asarray([row[3] for row in records])

    # observed (measured) flux is drawn in red so it stands out; the predicted
    # test split therefore uses green instead of the old red
    obs_color = "#d62728"
    colors = {"train": "C0", "val": "C1", "test": "C2"}
    indiv_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    date_lo, date_hi = dates[0], dates[-1]

    def style_axis(axis, combined):
        axis.set_yscale("log")
        axis.set_xlim(date_lo, date_hi)
        axis.set_ylabel(_flux_label(),
                        fontsize=9 if combined else 10)
        axis.xaxis.set_major_locator(
            mdates.AutoDateLocator(minticks=4 if combined else 5,
                                   maxticks=7 if combined else 9)
        )
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.tick_params(direction="in", top=True, right=True,
                         labelsize=8 if combined else 10)

    def bin_label(axis, j, combined):
        axis.text(
            0.02, 0.94, f"bin {j + 1}: {bins[j][0]:g}-{bins[j][1]:g} {PLOT_LABELS['unit']}",
            transform=axis.transAxes, ha="left", va="top",
            fontsize=11 if combined else 13, fontweight="bold",
        )

    # individual: detailed, one bin per figure, predicted split by colour
    for j in range(obs.shape[1]):
        figure, axis = plt.subplots(figsize=(11.0, 3.4))
        axis.plot(dates, obs[:, j], linestyle="none", marker=".",
                  markersize=2.6, color=obs_color, markeredgewidth=0, zorder=1,
                  label="AMS Data")
        for label in ("train", "val", "test"):
            sel = split_of == label
            if not sel.any():
                continue
            axis.plot(
                [d for d, s in zip(dates, sel) if s], prd[sel, j],
                linestyle="none", marker=".", markersize=2.4,
                color=colors[label], markeredgewidth=0, zorder=2,
                label=f"predicted {label}",
            )
        style_axis(axis, combined=False)
        bin_label(axis, j, combined=False)
        axis.legend(
            loc="upper left", bbox_to_anchor=(0.0, 0.88), borderaxespad=0.3,
            fontsize=9, frameon=False, markerscale=1.8, handlelength=1.2,
            labelcolor=[obs_color, "C0", "C1", "C2"],
        )
        figure.tight_layout()
        path = indiv_dir / f"bin_{j + 1:02d}_{bins[j][0]:g}-{bins[j][1]:g}{PLOT_LABELS['unit']}.pdf"
        figure.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        paths.append(str(path))

    # combined: simple and sparse, 6 bins per 3x2 figure, observed red + predicted blue
    n_bins = obs.shape[1]
    for group_start in range(0, n_bins, group_size):
        group = list(range(group_start, min(group_start + group_size, n_bins)))
        group_number = group_start // group_size + 1
        figure, axes = plt.subplots(3, 2, figsize=(16.0, 11.0), squeeze=False)
        for index, j in enumerate(group):
            axis = axes[index // 2, index % 2]
            axis.plot(dates, obs[:, j], linestyle="none", marker=".",
                      markersize=2.6, color=obs_color, markeredgewidth=0, zorder=1,
                      label="AMS Data")
            axis.plot(dates, prd[:, j], linestyle="none", marker=".",
                      markersize=2.4, color="#1a73e8", markeredgewidth=0, zorder=2,
                      label="predicted")
            style_axis(axis, combined=True)
            bin_label(axis, j, combined=True)
            axis.legend(
                loc="upper left", bbox_to_anchor=(0.0, 0.90), borderaxespad=0.3,
                fontsize=9, frameon=False, markerscale=2.2, handlelength=1.2,
                labelcolor=[obs_color, "#1a73e8"],
            )
        for index in range(len(group), group_size):
            axes[index // 2, index % 2].set_visible(False)
        figure.suptitle(
            f"{PLOT_LABELS['particle']} flux: AMS Data (red) vs predicted (blue) "
            f"(bins {group[0] + 1}-{group[-1] + 1})",
            fontsize=15, fontweight="bold",
        )
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        path = combined_dir / f"combined_timeseries_group_{group_number}.pdf"
        figure.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        paths.append(str(path))

    # observed only: the measured AMS flux alone, same 6-bins-per-figure grouping
    for group_start in range(0, n_bins, group_size):
        group = list(range(group_start, min(group_start + group_size, n_bins)))
        group_number = group_start // group_size + 1
        figure, axes = plt.subplots(3, 2, figsize=(16.0, 11.0), squeeze=False)
        for index, j in enumerate(group):
            axis = axes[index // 2, index % 2]
            axis.plot(dates, obs[:, j], linestyle="none", marker=".",
                      markersize=2.6, color=obs_color, markeredgewidth=0)
            style_axis(axis, combined=True)
            bin_label(axis, j, combined=True)
        for index in range(len(group), group_size):
            axes[index // 2, index % 2].set_visible(False)
        figure.suptitle(
            f"AMS Data: {PLOT_LABELS['particle']} flux (bins {group[0] + 1}-{group[-1] + 1})",
            fontsize=15, fontweight="bold",
        )
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        path = combined_dir / f"combined_observed_group_{group_number}.pdf"
        figure.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        paths.append(str(path))
    return paths


# --------------------------------------------------------------------------
# test-set diagnostics in the style of 小导's draw scripts (test samples only)
# --------------------------------------------------------------------------
def read_proton_total_error(meta_path: Path, n_rows: int):
    """AMS total error per (day, bin) = sqrt(stat^2 + syst^2), as 小导 uses.

    Returns None when the meta file carries no error arrays.
    """
    with np.load(meta_path) as meta:
        if not {"flux_err_statistical", "flux_err_systematic_total"} <= set(meta.files):
            return None
        stat = np.asarray(meta["flux_err_statistical"], dtype=np.float64)
        syst = np.asarray(meta["flux_err_systematic_total"], dtype=np.float64)
    if stat.shape[0] != n_rows:
        return None
    return np.sqrt(stat ** 2 + syst ** 2)


# Which AMS error the relative-difference figure uses (switch to compare):
#   "daily": sqrt(1^2 + 2^2) = stat (+) time-dependent syst -- the day-to-day
#            error AMS uses for its periodicity analysis; the time-independent
#            part cancels in pred/obs - 1 because the model learns it from AMS
#   "total": sqrt(1^2 + 3^2) = stat (+) total syst -- the full error of a point;
#            numerically what Zhao et al. 2026 Fig. 2 (bottom, red) shows
RELERR_AMS_ERR = "total"
RELERR_AMS_ERR_COLUMNS = {
    "daily": "flux_err_timedependent",
    "total": "flux_err_systematic_total",
}
RELERR_AMS_ERR_LABELS = {
    "daily": "AMS stat$\\oplus$time-dep. syst (mean)",
    "total": "AMS stat$\\oplus$syst (mean)",
}


def read_proton_daily_relerr(meta_path: Path, rows=None, kind=None):
    """AMS relative error for every (day, bin), shape (n_days, n_bins):
    sqrt(stat^2 + syst^2) / flux, where syst is the time-dependent column
    (kind "daily") or the total systematic column (kind "total"); kind None
    uses RELERR_AMS_ERR.

    ``rows`` selects the days (row indices of the proton meta), e.g. the test
    split, so it describes the same days as the plotted predictions; None uses
    every day. Returns None when the meta file lacks the needed columns.
    """
    kind = RELERR_AMS_ERR if kind is None else kind
    syst_col = RELERR_AMS_ERR_COLUMNS[kind]
    with np.load(meta_path) as meta:
        need = {"flux", "flux_err_statistical", syst_col}
        if not need <= set(meta.files):
            return None
        flux = np.asarray(meta["flux"], dtype=np.float64)
        stat = np.asarray(meta["flux_err_statistical"], dtype=np.float64)
        syst = np.asarray(meta[syst_col], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.sqrt(stat ** 2 + syst ** 2) / flux
    if rows is not None:
        rel = rel[np.asarray(rows)]
    return rel


def read_proton_daily_relerr_band(meta_path: Path, rows=None, kind=None):
    """Per-bin mean over the selected days of read_proton_daily_relerr."""
    rel = read_proton_daily_relerr(meta_path, rows=rows, kind=kind)
    return None if rel is None else np.nanmean(rel, axis=0)


def read_proton_relerr_parts(meta_path: Path, rows=None):
    """Per-bin mean over the selected days of the AMS error components, as
    relative errors: {"stat": stat/flux, "syst": total syst/flux,
    "daily": sqrt(stat^2 + time-dependent syst^2)/flux}. Used for the separate
    statistical / systematic bands of the relative-difference figure. Returns
    None when the meta file lacks the columns."""
    cols = ("flux", "flux_err_statistical", "flux_err_systematic_total",
            "flux_err_timedependent")
    with np.load(meta_path) as meta:
        if not set(cols) <= set(meta.files):
            return None
        flux, stat, syst, tdep = (np.asarray(meta[c], dtype=np.float64) for c in cols)
    if rows is not None:
        rows = np.asarray(rows)
        flux, stat, syst, tdep = flux[rows], stat[rows], syst[rows], tdep[rows]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "stat": np.nanmean(stat / flux, axis=0),
            "syst": np.nanmean(syst / flux, axis=0),
            "daily": np.nanmean(np.sqrt(stat ** 2 + tdep ** 2) / flux, axis=0),
        }


def match_proton_rows(meta_path: Path, flux_rows: np.ndarray) -> np.ndarray:
    """Row index in the proton meta of each given flux row (e.g. a saved
    *_test_true_flux.npy). Raises if any row has no exact match, so a band is
    never silently computed on the wrong days."""
    with np.load(meta_path) as meta:
        flux = np.asarray(meta["flux"], dtype=np.float64)
    lookup = {row.tobytes(): i for i, row in enumerate(flux)}
    rows = []
    for k, row in enumerate(np.asarray(flux_rows, dtype=np.float64)):
        i = lookup.get(row.tobytes())
        if i is None:
            close = np.where(np.all(np.isclose(flux, row, rtol=1e-9, atol=0), axis=1))[0]
            if close.size != 1:
                raise ValueError(f"flux row {k} not found in {meta_path}")
            i = int(close[0])
        rows.append(i)
    return np.asarray(rows)


def make_test_diagnostic_plots(
    out_dir, test_dates, y_true, y_pred, y_err, bins,
    panel_sets=((0, 6, 12, 18), (2, 8, 14, 20)),
    relerr_band=None,
    relerr_daily=None,
    relerr_label=None,
    relerr_parts=None,
):
    """Three figure types from 小导's scripts, drawn on the test set only.

    1) test_relerr_timeseries/bin_XX_*.pdf  -- per bin, top: observed vs
       predicted flux over time; bottom: signed relative difference
       (pred/obs - 1) over time          [draw_n2p_residual_*: seed_N_test_error]
    2) test_panels_bins_<a>_<b>_<c>_<d>.pdf -- 4 stacked panels per set
       (bins 1/7/13/19 as 小导, plus 3/9/15/21): AMS Data with error bars +
       prediction, both test samples only, same marker size
                                          [draw_n2p_residual_interpolate: n2p_seed_N]
    3) test_relerr_vs_rigidity.pdf       -- 2-D histogram of the signed
       relative difference vs rigidity, with the per-bin mean overlaid
                                          [draw_error_n2p_residual_interpolate]
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    order = np.argsort(np.asarray(test_dates))
    dates = [test_dates[i] for i in order]
    obs = np.asarray(y_true)[order]
    prd = np.asarray(y_pred)[order]
    err = np.asarray(y_err)[order] if y_err is not None else None
    rel = prd / obs - 1.0                              # signed, like 小导
    n_bins = obs.shape[1]
    obs_color, pred_color = "#d62728", "#1a73e8"
    flux_label = _flux_label()
    paths: list[str] = []

    def year_axis(axis):
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.tick_params(direction="in", top=True, right=True)

    # 1) per bin: flux + signed relative difference vs time
    rel_dir = out_dir / "test_relerr_timeseries"
    rel_dir.mkdir(parents=True, exist_ok=True)
    for j in range(n_bins):
        figure, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12.0, 7.0), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
        ax1.plot(dates, obs[:, j], linestyle="none", marker=".", markersize=4,
                 color=obs_color, markeredgewidth=0, label="AMS Data (test)")
        ax1.plot(dates, prd[:, j], linestyle="none", marker=".", markersize=3,
                 color=pred_color, markeredgewidth=0, label="predicted (test)")
        ax1.set_ylabel(flux_label)
        ax1.set_title(f"{PLOT_LABELS['particle']}, [{bins[j][0]:g}, {bins[j][1]:g}] "
                      f"{PLOT_LABELS['unit']}  (test set)",
                      fontsize=13, fontweight="bold")
        ax1.legend(loc="best", frameon=False, markerscale=2.5)
        ax2.axhline(0.0, color="0.4", linewidth=0.8)
        ax2.plot(dates, rel[:, j], linestyle="none", marker=".", markersize=4,
                 color=pred_color, markeredgewidth=0)
        ax2.set_ylabel("pred / obs - 1")
        ax2.set_xlabel("year")
        ax2.text(0.01, 0.95,
                 f"mean {np.mean(rel[:, j]) * 100:+.2f}%   "
                 f"mean |.| {np.mean(np.abs(rel[:, j])) * 100:.2f}%",
                 transform=ax2.transAxes, ha="left", va="top", fontsize=9)
        for axis in (ax1, ax2):
            year_axis(axis)
        figure.tight_layout()
        path = rel_dir / f"bin_{j + 1:02d}_{bins[j][0]:g}-{bins[j][1]:g}{PLOT_LABELS['unit']}.pdf"
        figure.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        paths.append(str(path))

    # 2) stacked panels with AMS error bars, one figure per bin set
    for panel_bins in panel_sets:
      chosen = [j for j in panel_bins if j < n_bins]
      if chosen:
          figure, axes = plt.subplots(len(chosen), 1, figsize=(10.0, 2.6 * len(chosen)),
                                      sharex=True, squeeze=False)
          for axis, j in zip(axes[:, 0], chosen):
              if err is not None:
                  axis.errorbar(dates, obs[:, j], yerr=err[:, j], fmt="o",
                                markersize=2.5, markeredgewidth=0,
                                color=obs_color, ecolor=obs_color,
                                elinewidth=0.6, alpha=0.8,
                                label=f"AMS Data (test), {PLOT_LABELS['err']}")
              else:
                  axis.plot(dates, obs[:, j], linestyle="none", marker="o",
                            markersize=2.5, markeredgewidth=0,
                            color=obs_color, label="AMS Data (test)")
              axis.plot(dates, prd[:, j], linestyle="none", marker="o", markersize=2.5,
                        color=pred_color, markeredgewidth=0,
                        label="model prediction (test)")
              axis.text(0.99, 0.06, f"[{bins[j][0]:g}, {bins[j][1]:g}] {PLOT_LABELS['unit']}",
                        transform=axis.transAxes, ha="right", fontsize=10,
                        fontweight="bold")
              year_axis(axis)
          axes[0, 0].legend(loc="upper left", frameon=False, fontsize=9, markerscale=2)
          axes[-1, 0].set_xlabel("year")
          figure.supylabel(PLOT_LABELS["particle"] + " " + flux_label, fontsize=11)
          figure.tight_layout()
          tag = "_".join(str(j + 1) for j in chosen)
          path = out_dir / f"test_panels_bins_{tag}.pdf"
          figure.savefig(path, format="pdf", bbox_inches="tight")
          plt.close(figure)
          paths.append(str(path))

    # 3) 2-D histogram: signed relative difference vs rigidity
    paths.append(plot_relerr_vs_rigidity(out_dir, rel, bins, err_band=relerr_band,
                                         err_daily=relerr_daily,
                                         err_label=relerr_label,
                                         err_parts=relerr_parts))
    return paths


# AMS error components drawn as light-red lines (err_parts of
# plot_relerr_vs_rigidity): statistical dashed, total systematic dotted
PARTS_STYLE = {
    "stat": {"color": "#f08a8a", "ls": "--", "label": "AMS stat (mean)"},
    "syst": {"color": "#f4b6b6", "ls": ":", "label": "AMS syst (mean)"},
}


# y range of the relative-difference 2-D histogram (points outside are not
# drawn; the red per-bin mean still uses every test day)
RELERR_YLIM = 0.1


def plot_relerr_vs_rigidity(out_dir, rel, bins, ylim=None, err_band=None,
                            show_std=None, err_daily=None, ams_err_kind=None,
                            err_label=None, err_parts=None):
    """Relative difference (pred/obs - 1) vs rigidity, after Zhao et al. 2026
    (ApJS 282:5) Fig. 2.

    Top panel: 2-D histogram of pred/obs - 1 over the plotted (test) days,
    fixed y range +-ylim, with per bin
      * blue points: the mean (overall bias of the prediction);
      * blue steps:  mean +- standard deviation (the spread of the prediction,
                     centred on its own mean so it matches the histogram);
      * red steps:   +- the AMS day-to-day relative error (mean over the days),
                     centred on 0.
    Bottom panel (only with ``err_daily``): 2-D histogram of the daily AMS
    relative error sqrt(stat^2 + time-dependent syst^2)/flux of the same days,
    its per-bin mean (red points) and the std of pred/obs - 1 (blue points,
    taken as the model's uncertainty by Zhao et al.), so the two widths can be
    compared directly.

    ``err_parts`` (optional, protons): read_proton_relerr_parts of the plotted
    days; adds two light-red step lines, +- AMS stat (dashed) and +- AMS total
    syst (dotted), centred on 0 to the top panel, and the same two as lines
    (mean per bin) to the bottom panel (no distribution of them is drawn).

    ``err_daily``: (n_days, n_bins) daily AMS relative error of the plotted days
    (read_proton_daily_relerr with the test rows); ``err_band`` defaults to its
    per-bin mean. Callers without AMS errors (e.g. nmdb_p2pos.py) pass neither
    and get the top panel with the mean only, as before. ``show_std`` defaults
    to "an AMS error is given".
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import MultipleLocator

    red, blue = "#d62728", "#1a73e8"
    # which AMS error err_daily/err_band hold (legend text only)
    ams_err_kind = RELERR_AMS_ERR if ams_err_kind is None else ams_err_kind
    # legend of the AMS error; err_label overrides it for data whose error is
    # not the proton stat/syst split (e.g. the single positron error column)
    err_text = err_label if err_label is not None else RELERR_AMS_ERR_LABELS[ams_err_kind]
    ylim = RELERR_YLIM if ylim is None else float(ylim)
    rel = np.asarray(rel, dtype=np.float64)
    if err_daily is not None:
        err_daily = np.asarray(err_daily, dtype=np.float64)
        if err_band is None:
            err_band = np.nanmean(err_daily, axis=0)
    if show_std is None:
        show_std = err_band is not None
    edges = np.asarray([bins[0][0]] + [b[1] for b in bins], dtype=np.float64)
    centers = np.sqrt(edges[:-1] * edges[1:])
    mean = rel.mean(axis=0)
    std = rel.std(axis=0)          # every plotted day, also those outside +-ylim

    def steps(values):             # step outline over the bin edges
        values = np.asarray(values, dtype=np.float64)
        return np.append(values, values[-1])

    def hist2d(axis, values, y_edges, label):
        x_vals = np.repeat(centers[None, :], values.shape[0], axis=0).ravel()
        flat = values.ravel()
        ok = np.isfinite(flat)
        counts, _, _ = np.histogram2d(x_vals[ok], flat[ok], bins=[edges, y_edges])
        masked = np.ma.masked_where(counts.T == 0, counts.T)
        mesh = axis.pcolormesh(edges, y_edges, masked, cmap="viridis",
                               norm=LogNorm(vmin=1, vmax=max(1, counts.max())))
        axis.figure.colorbar(mesh, ax=axis, label=label)

    n_out = int(np.sum(np.abs(rel) > ylim))

    def draw(layout):
        """layout "tall": panels stacked (paper); "wide": side by side (slides)."""
        if err_daily is None:
            figure, axis = plt.subplots(figsize=(9.0, 6.0))
            ax_lo = None
        elif layout == "tall":
            figure, (axis, ax_lo) = plt.subplots(
                2, 1, figsize=(9.0, 9.5), sharex=True,
                gridspec_kw={"height_ratios": [2.0, 1.2]},
            )
        else:
            figure, (axis, ax_lo) = plt.subplots(
                1, 2, figsize=(16.0, 6.4),
                gridspec_kw={"width_ratios": [1.0, 1.0]},
            )
        xlabel = f"{PLOT_LABELS['axis']} ({PLOT_LABELS['unit']})"

        # ---- relative difference ----------------------------------------
        # 40 rows over +-ylim (row height ylim/20): 0.005 for protons at
        # ylim 0.1, like 小导; wider rows for the noisier positrons
        hist2d(axis, rel, np.linspace(-ylim, ylim, 41), "test days")
        if err_band is not None:
            band = steps(err_band)
            axis.step(edges, band, where="post", color=red, linewidth=1.6, zorder=3,
                      label=err_text)
            axis.step(edges, -band, where="post", color=red, linewidth=1.6, zorder=3)
        if err_parts is not None:
            # thin light-red lines only (no fill), so the histogram stays visible
            for key, style in PARTS_STYLE.items():
                band = steps(err_parts[key])
                axis.step(edges, band, where="post", color=style["color"],
                          linestyle=style["ls"], linewidth=1.3, zorder=3,
                          label=style["label"])
                axis.step(edges, -band, where="post", color=style["color"],
                          linestyle=style["ls"], linewidth=1.3, zorder=3)
        if show_std:
            axis.step(edges, steps(mean + std), where="post", color=blue,
                      linewidth=1.6, zorder=3,
                      label="mean $\\pm$ std of relative difference")
            axis.step(edges, steps(mean - std), where="post", color=blue,
                      linewidth=1.6, zorder=3)
        axis.plot(centers, mean, "o--", color=blue, markersize=4,
                  linewidth=1.2, label="mean per bin", zorder=4)
        axis.axhline(0.0, color="0.3", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_ylim(-ylim, ylim)
        axis.yaxis.set_major_locator(MultipleLocator(ylim / 10.0))
        axis.set_ylabel("relative difference  (pred / obs - 1)")
        axis.set_title(f"test set: relative difference vs {PLOT_LABELS['axis']}")
        axis.legend(loc="lower right", frameon=True, facecolor="white",
                    framealpha=0.8, edgecolor="none")
        axis.tick_params(direction="in", top=True, right=True, which="both")

        # ---- relative uncertainties -------------------------------------
        if ax_lo is not None:
            finite = err_daily[np.isfinite(err_daily)]
            # own range, not capped by the top panel: every red/blue point fits
            top = max(float(np.max(std)), float(np.nanmax(err_band)),
                      float(np.percentile(finite, 99.5)) if finite.size else 0.0)
            top = max(top * 1.15, 0.01)
            hist2d(ax_lo, err_daily, np.linspace(0.0, top, 61), "test days")
            ax_lo.plot(centers, err_band, "o", color=red, markersize=4, zorder=4,
                       label=err_text)
            if err_parts is not None:
                for key, style in PARTS_STYLE.items():
                    ax_lo.plot(centers, err_parts[key], marker="o", markersize=3,
                               color=style["color"], linestyle=style["ls"],
                               linewidth=1.3, zorder=4, label=style["label"])
            ax_lo.plot(centers, std, "o-", color=blue, markersize=4, linewidth=1.2,
                       zorder=4, label="std of relative difference (model)")
            ax_lo.set_xscale("log")
            ax_lo.set_xlim(edges[0], edges[-1])
            ax_lo.set_ylim(0.0, top)
            ax_lo.set_ylabel("relative uncertainty")
            ax_lo.legend(loc="upper right", frameon=True, facecolor="white",
                         framealpha=0.8, edgecolor="none")
            ax_lo.tick_params(direction="in", top=True, right=True, which="both")
            ax_lo.set_xlabel(xlabel)
            if layout == "wide":
                axis.set_xlabel(xlabel)
                ax_lo.set_title(f"test set: relative uncertainties vs "
                                f"{PLOT_LABELS['axis']}")
        else:
            axis.set_xlabel(xlabel)

        figure.tight_layout()
        name = ("test_relerr_vs_rigidity.pdf" if layout == "tall"
                else "test_relerr_vs_rigidity_wide.pdf")
        path = Path(out_dir) / name
        figure.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        return path

    path = draw("tall")
    if err_daily is not None:
        # same content side by side, 16:9-ish, for slides
        print(f"relerr figure (wide, for slides): {draw('wide')}")
    print(f"relerr 2-D histogram: {n_out} of {rel.size} (day, bin) points "
          f"outside +-{ylim:g} not drawn")
    return str(path)


# --------------------------------------------------------------------------
# --select-seed: compare the seed*/ runs of one group and pick the best
# --------------------------------------------------------------------------
# 小导 trains one model per data_seed and keeps the one with the lowest
# best val loss (the early-stopping value, MSE + L2 on the scaled target).
# The test set is not used to choose, so the chosen run's test numbers stay
# unbiased. "--select-by test_mre" exists only for comparison and is flagged.
SELECT_LOWBIN = 5                       # bins 1..5 (~1-1.9 GV) for the 2017 bias
SELECT_WINDOW = ("2017-07-01", "2017-11-01")


def _sel_split_indices(n, test_frac, val_frac, data_seed):
    """Same two-step split as nmdb_n2p.py (sklearn ShuffleSplit semantics)."""
    def once(idx, frac):
        n_test = math.ceil(frac * len(idx))
        perm = np.random.RandomState(data_seed).permutation(len(idx))
        return idx[perm[n_test:]], idx[perm[:n_test]]

    train_idx, test_idx = once(np.arange(n), test_frac)
    train_idx, val_idx = once(train_idx, val_frac / (1.0 - test_frac))
    return train_idx, val_idx, test_idx


def _sel_one_file(folder: Path, pattern: str):
    hits = sorted(folder.glob(pattern))
    return hits[0] if len(hits) == 1 else None


def _sel_window_bias(rel, dates, lo, hi):
    sel = (dates >= np.datetime64(lo)) & (dates < np.datetime64(hi))
    return (float(rel[sel, :SELECT_LOWBIN].mean()), int(sel.sum())) if sel.any() else (float("nan"), 0)


def _sel_collect(n2p_dir: Path, proton_dates, summary_name="n2p_summary.json"):
    rows = []
    for summary_path in sorted(n2p_dir.glob(f"seed*/{summary_name}")):
        run = summary_path.parent
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        m = s["metrics"]
        row = {
            "seed": int(s["data_seed"]),
            "run_dir": str(run),
            "best_val_loss": float(s["best_val_loss"]) if s.get("best_val_loss") is not None else float("nan"),
            "best_epoch": s.get("best_epoch"),
            "epochs_run": s.get("epochs_run"),
            "train_mre": m["train"]["mean_abs_relative_error"],
            "val_mre": m["val"]["mean_abs_relative_error"],
            "test_mre": m["test"]["mean_abs_relative_error"],
        }
        err = run / "Error"
        rel_test = _sel_one_file(err, "*_test_rel_error.npy")
        rel_train = _sel_one_file(err, "*_train_rel_error.npy")
        if rel_test is not None and rel_train is not None and proton_dates is not None:
            rt, rr = np.load(rel_test), np.load(rel_train)
            row["test_mre_bin1"] = float(np.abs(rt[:, 0]).mean())
            row["test_bias"] = float(rt.mean())
            tr, _, te = _sel_split_indices(len(proton_dates), s["test_frac"], s["val_frac"], s["data_seed"])
            if len(te) == len(rt) and len(tr) == len(rr):
                row["train_bias_2017_low"], _ = _sel_window_bias(rr, proton_dates[tr], *SELECT_WINDOW)
                row["test_bias_2017_low"], row["n_test_2017"] = _sel_window_bias(rt, proton_dates[te], *SELECT_WINDOW)
        rows.append(row)
    return rows



def select_seed(n2p_dir: Path, proton_meta, by: str = "val_loss",
                summary_name: str = "n2p_summary.json", label: str = "n2p") -> int:
    """Write seed_summary.csv, best_seed.json, seed_selection.pdf into n2p_dir.

    Also used by nmdb_p2pos.py (summary_name="p2pos_summary.json",
    proton_meta=None): the 2017 low-rigidity proton columns are then left empty.
    """
    n2p_dir = Path(n2p_dir)
    group = n2p_dir.name.removeprefix("nmdb_n2p_").removeprefix("nmdb_p2pos_")
    proton_dates = None
    if proton_meta is not None and Path(proton_meta).is_file():
        with np.load(proton_meta, allow_pickle=True) as meta:
            proton_dates = np.asarray(meta["dates"]).astype("datetime64[D]")
    rows = _sel_collect(n2p_dir, proton_dates, summary_name)
    if not rows:
        raise SystemExit(f"no seed*/{summary_name} under {n2p_dir}")

    key = {"val_loss": "best_val_loss", "val_mre": "val_mre", "test_mre": "test_mre"}[by]
    vals = np.array([r[key] for r in rows], dtype=float)
    if np.all(np.isnan(vals)):
        raise SystemExit(f"no run has {key}; re-run with the current nmdb_n2p.py or use --by val_mre")
    best = rows[int(np.nanargmin(vals))]
    rows.sort(key=lambda r: (math.isnan(r[key]), r[key]))

    cols = ["seed", "best_val_loss", "best_epoch", "epochs_run", "train_mre", "val_mre",
            "test_mre", "test_mre_bin1", "test_bias", "train_bias_2017_low",
            "test_bias_2017_low", "n_test_2017", "run_dir"]
    with (n2p_dir / "seed_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    test = np.array([r["test_mre"] for r in rows])
    spread = {
        "n_seeds": len(rows),
        "test_mre_mean": float(test.mean()), "test_mre_std": float(test.std(ddof=1)) if len(test) > 1 else 0.0,
        "test_mre_min": float(test.min()), "test_mre_max": float(test.max()),
    }
    out = {
        "group": group, "criterion": by,
        "note": "test set used for selection -> optimistic" if by == "test_mre" else "test set not used for selection",
        "best_seed": best["seed"], "best_run_dir": best["run_dir"],
        "best": {k: best.get(k) for k in cols if k != "run_dir"},
        "all_seeds": spread,
        "missing_best_val_loss": [r["seed"] for r in rows if math.isnan(r["best_val_loss"])],
    }
    (n2p_dir / "best_seed.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'seed':>4} {'val_loss':>10} {'test MRE':>9} {'bin1':>7} {'2017 tr':>8} {'2017 te':>8}")
    for r in rows:
        f = lambda k, s=100: (f"{r[k]*s:+.2f}%" if k.endswith('bias_2017_low') else f"{r[k]*s:.2f}%") if isinstance(r.get(k), float) and not math.isnan(r[k]) else "-"
        mark = "  <- best" if r is best else ""
        print(f"{r['seed']:>4} {r['best_val_loss']:>10.6f} {f('test_mre'):>9} {f('test_mre_bin1'):>7} "
              f"{f('train_bias_2017_low'):>8} {f('test_bias_2017_low'):>8}{mark}")
    print(f"test MRE over {len(rows)} seeds: {100*spread['test_mre_mean']:.3f}% +- {100*spread['test_mre_std']:.3f}% "
          f"(min {100*spread['test_mre_min']:.3f}%, max {100*spread['test_mre_max']:.3f}%)")
    print(f"best by {by}: seed{best['seed']}  -> {n2p_dir / 'best_seed.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        by_seed = sorted(rows, key=lambda r: r["seed"])
        seeds = [r["seed"] for r in by_seed]
        tm = np.array([r["test_mre"] for r in by_seed]) * 100
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        a1.plot(seeds, tm, "o", color="#1a73e8", markersize=5)
        a1.plot([best["seed"]], [best["test_mre"] * 100], "o", color="red", markersize=8,
                label=f"chosen (by {by}): seed {best['seed']}")
        a1.axhline(tm.mean(), color="0.4", linewidth=1, label=f"mean {tm.mean():.3f}%")
        if len(tm) > 1:
            a1.axhspan(tm.mean() - tm.std(ddof=1), tm.mean() + tm.std(ddof=1), color="0.85", label="mean +- std")
        a1.set_xlabel("data_seed"); a1.set_ylabel("test MRE (%)")
        a1.legend(frameon=False, fontsize=9)
        vl = np.array([r["best_val_loss"] for r in by_seed])
        ok = ~np.isnan(vl)
        a2.plot(vl[ok], tm[ok], "o", color="#1a73e8", markersize=5)
        if not math.isnan(best["best_val_loss"]):
            a2.plot([best["best_val_loss"]], [best["test_mre"] * 100], "o", color="red", markersize=8)
        for s_, x_, y_ in zip(np.array(seeds)[ok], vl[ok], tm[ok]):
            a2.annotate(str(s_), (x_, y_), fontsize=7, xytext=(3, 3), textcoords="offset points")
        a2.set_xlabel("best val loss"); a2.set_ylabel("test MRE (%)")
        for a in (a1, a2):
            a.tick_params(direction="in", top=True, right=True)
        fig.suptitle(f"{label} {group}: {len(rows)} data_seeds")
        fig.tight_layout()
        fig.savefig(n2p_dir / "seed_selection.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # plotting is optional
        print(f"[skip plot] {exc}")
    return 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent          # .../LeptonFluxPrediction/NM
    proton_dir = base.parent / "PROTON"
    parser = argparse.ArgumentParser(
        description="Train the n2p residual network (neutron -> proton), PyTorch."
    )
    parser.add_argument(
        "--neutron", type=Path,
        default=base / "data" / "nmdb_imputed" / "nm_daily_imputed_counts.npy",
        help="SAITS-imputed NM daily matrix (T, n_stations), fully finite",
    )
    parser.add_argument(
        "--neutron-start", default=None,
        help=(
            "date of the first row of the neutron matrix; only needed when no "
            "dated CSV sits next to --neutron (the CSV date column wins, and a "
            "mismatch is an error)"
        ),
    )
    parser.add_argument(
        "--proton", type=Path,
        default=proton_dir / "data" / "proton_allbin_dropna.npy",
        help="AMS daily proton flux (dropna, complete), (T, n_bins)",
    )
    parser.add_argument(
        "--proton-meta", type=Path,
        default=proton_dir / "data" / "proton_allbin_dropna_meta.npz",
        help="meta npz with the dates matching --proton",
    )
    parser.add_argument(
        "--stations", default=None,
        help=(
            "comma-separated station subset (default: all 18, Rc order); "
            "matrix column order is always kept"
        ),
    )
    parser.add_argument("--data-seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument(
        "--init", choices=("keras", "torch"), default="keras",
        help="weight init: 'keras' = glorot_uniform + zero bias as 小导's Keras "
             "code (default); 'torch' = PyTorch default (Kaiming-uniform)",
    )
    parser.add_argument(
        "--drop-last", action="store_true",
        help="drop the last incomplete training batch each epoch (old behaviour); "
             "default keeps it, as Keras model.fit does",
    )
    parser.add_argument(
        "--device", default=None,
        help="torch device (default: cuda > mps > cpu)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=base / "data" / "nmdb_n2p",
        help=(
            "root for Model/ Figure/ Error/ outputs and n2p_summary.json "
            "(default: NM/data/nmdb_n2p; the group jobs use "
            "data/nmdb_n2p_<TAG>/seed<N>, one directory per run because the "
            "summary file name is fixed)"
        ),
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="fast smoke run: epochs<=500 and write into <output-dir>/quick",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--no-log", action="store_true",
        help="target = raw flux (per-bin MinMax only, no log10); default is "
             "log10 + MinMax as 小导. Output file tag gets a '_nolog' suffix",
    )
    parser.add_argument(
        "--select-seed", action="store_true",
        help="no training: compare all <output-dir>/seed*/ runs of one group, "
             "write seed_summary.csv, best_seed.json and seed_selection.pdf "
             "into <output-dir> (e.g. --output-dir data/nmdb_n2p_lowrc9lj)",
    )
    parser.add_argument(
        "--select-by", choices=("val_loss", "val_mre", "test_mre"), default="val_loss",
        help="criterion for --select-seed (default val_loss, as 小导; "
             "test_mre uses the test set to choose -> optimistic)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="global seed for weight init etc."
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.select_seed:
        return select_seed(args.output_dir, args.proton_meta, args.select_by)

    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        import torch
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler
    except ImportError as exc:
        raise SystemExit(
            f"missing dependency: {exc}\n"
            "this script needs torch and scikit-learn in the run env"
        ) from exc
    torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    if args.quick:
        args.epochs = min(args.epochs, 500)
        args.output_dir = args.output_dir / "quick"

    start_date = (
        dt.date.fromisoformat(args.neutron_start) if args.neutron_start else None
    )
    neutron, neutron_dates = read_neutron(args.neutron, start_date)
    proton, proton_dates = read_proton(args.proton, args.proton_meta)
    rows = align_rows(neutron_dates, proton_dates)

    station_names = read_neutron_station_names(args.neutron, neutron.shape[1])
    indices = select_station_indices(station_names, args.stations)
    used_stations = [station_names[i] for i in indices]
    X = neutron[rows][:, indices]
    y_raw = proton
    n_days, n_stations = X.shape
    n_bins = y_raw.shape[1]
    if n_stations < 2:
        raise SystemExit(
            f"need at least 2 stations for the conv residual model, got {n_stations}"
        )
    print(f"aligned days={n_days} stations={n_stations} bins={n_bins}")
    print("stations:", ", ".join(used_stations))

    y_is_log10 = not args.no_log
    # training target: log10(flux) (default) or the raw flux (--no-log);
    # the name y_log is kept below, it holds whichever target is used
    y_log = np.log10(y_raw) if y_is_log10 else np.asarray(y_raw, dtype=np.float64)
    print("target:", "log10(flux) + MinMax" if y_is_log10 else "raw flux + MinMax (no log)")

    # split by index so the dates of each split are kept for the time-series plots
    all_idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=args.test_frac, shuffle=True,
        random_state=args.data_seed,
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=args.val_frac / (1.0 - args.test_frac), shuffle=True,
        random_state=args.data_seed,
    )
    X_train, y_train = X[train_idx], y_log[train_idx]
    X_val, y_val = X[val_idx], y_log[val_idx]
    X_test, y_test = X[test_idx], y_log[test_idx]
    proton_dates_arr = np.asarray(proton_dates)
    date_splits = {
        "train": proton_dates_arr[train_idx],
        "val": proton_dates_arr[val_idx],
        "test": proton_dates_arr[test_idx],
    }

    scaler_x = MinMaxScaler().fit(X_train)
    X_train = scaler_x.transform(X_train)[..., None]     # (n, N, 1)
    X_val = scaler_x.transform(X_val)[..., None]
    X_test = scaler_x.transform(X_test)[..., None]
    scaler_y = MinMaxScaler().fit(y_train)
    y_train_s = scaler_y.transform(y_train)
    y_val_s = scaler_y.transform(y_val)
    y_test_s = scaler_y.transform(y_test)

    model, kernels = build_model(n_stations, n_bins, args.dropout, init=args.init)
    model.to(device)
    print("kernels:", kernels)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params:,}")

    # L2 exactly as Keras L2(l2) on the Dense kernel: + l2 * sum(W**2) on the
    # output weight only (not its bias), added to the loss itself. The optimizer
    # therefore gets no weight_decay; the penalty is also included in the
    # val/test loss, as Keras' val_loss does, so early stopping matches.
    optimizer = torch.optim.Adamax(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-7,
    )
    mse = torch.nn.MSELoss()

    def criterion(pred, target):
        return mse(pred, target) + args.l2 * torch.sum(model.head.weight ** 2)

    X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train_s, dtype=torch.float32)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val_s, dtype=torch.float32, device=device)
    X_test_t = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.as_tensor(y_test_s, dtype=torch.float32, device=device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_t, y_train_t),
        batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last,
    )

    model_dir = args.output_dir / "Model"
    fig_dir = args.output_dir / "Figure"
    err_dir = args.output_dir / "Error"
    for path in (model_dir, fig_dir, err_dir):
        path.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seed{args.data_seed}_{n_stations}st_"
        f"{args.epochs}ep_lr{args.lr}_l2{args.l2}_do{args.dropout}_b{args.batch_size}"
    ) + ("" if y_is_log10 else "_nolog")
    ckpt_path = model_dir / f"{tag}_best.pt"
    # the two train-split scalers, so nmdb_n2p_extend.py can reuse them as-is
    scaler_path = model_dir / "scalers.npz"
    np.savez(
        scaler_path,
        stations=np.asarray(used_stations),
        x_data_min=scaler_x.data_min_, x_data_max=scaler_x.data_max_,
        y_data_min=scaler_y.data_min_, y_data_max=scaler_y.data_max_,
        y_is_log10=np.asarray(y_is_log10),
    )
    # same coefficients as a table you can open directly (Excel / text editor)
    scaler_csv_path = model_dir / "scalers.csv"
    write_scalers_csv(
        scaler_csv_path, used_stations, scaler_x.data_min_, scaler_x.data_max_,
        scaler_y.data_min_, scaler_y.data_max_, read_bins(args.proton_meta), y_is_log10,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    test_losses: list[float] = []
    best_val = float("inf")
    best_epoch = -1
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        train_loss = float(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(X_val_t), y_val_t).item())
            test_loss = float(criterion(model(X_test_t), y_test_t).item())
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        improved = val_loss < best_val - args.min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            patience_left = args.patience
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_left -= 1

        if epoch == 1 or epoch % 10 == 0 or improved or patience_left == 0:
            print(
                f"epoch {epoch:5d} train {train_loss:.6f} "
                f"val {val_loss:.6f} test {test_loss:.6f}"
                + (" *" if improved else "")
            )

        if patience_left == 0:
            print(
                f"early stopping at epoch {epoch}; best val {best_val:.6f} "
                f"@ epoch {best_epoch}"
            )
            break

    if ckpt_path.is_file():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"loaded best checkpoint: {ckpt_path.name} (epoch {best_epoch})")

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
    metrics: dict[str, dict[str, float]] = {}
    true_by_split: dict[str, np.ndarray] = {}
    pred_by_split: dict[str, np.ndarray] = {}
    for name, (X_in, y_log_true) in splits.items():
        true_flux = np.power(10.0, y_log_true) if y_is_log10 else np.asarray(y_log_true)
        pred_flux = predict_flux(model, X_in, scaler_y, device, y_is_log10)
        true_by_split[name] = true_flux
        pred_by_split[name] = pred_flux
        np.save(err_dir / f"{tag}_{name}_true_flux.npy", true_flux)
        np.save(err_dir / f"{tag}_{name}_pred_flux.npy", pred_flux)
        # signed relative difference per (day, bin), as 小导 saves it
        np.save(err_dir / f"{tag}_{name}_rel_error.npy", pred_flux / true_flux - 1.0)
        # log10 metrics on positive predictions only (a raw-flux target can
        # give pred <= 0); n_nonpositive_pred reports how many were left out
        positive = pred_flux > 0
        dlog = np.log10(pred_flux[positive]) - np.log10(true_flux[positive])
        metrics[name] = {
            "mse_log10": float(np.mean(dlog ** 2)) if dlog.size else None,
            "mae_log10": float(np.mean(np.abs(dlog))) if dlog.size else None,
            "n_nonpositive_pred": int((~positive).sum()),
            "mse_physical": float(np.mean((pred_flux - true_flux) ** 2)),
            "mae_physical": float(np.mean(np.abs(pred_flux - true_flux))),
            "mean_abs_relative_error": float(
                np.mean(np.abs(pred_flux - true_flux) / true_flux)
            ),
        }
        print(name, json.dumps(metrics[name]))

    plot_paths: list[str] = []
    ts_paths: list[str] = []
    if not args.no_plots:
        test_true = true_by_split["test"]
        test_pred = pred_by_split["test"]
        bins = read_bins(args.proton_meta)
        try:
            plot_paths = make_plots(
                fig_dir, train_losses, val_losses, test_losses,
                test_true, test_pred, bins,
            )
        except Exception as exc:
            print(f"[skip plots] {exc}")
        try:
            ts_paths = make_timeseries_plots(
                fig_dir / "timeseries", fig_dir, date_splits,
                true_by_split, pred_by_split, bins,
            )
        except Exception as exc:
            print(f"[skip timeseries plots] {exc}")
        try:
            total_err = read_proton_total_error(args.proton_meta, len(proton_dates))
            test_err = total_err[test_idx] if total_err is not None else None
            diag_paths = make_test_diagnostic_plots(
                fig_dir, list(date_splits["test"]), test_true, test_pred,
                test_err, bins,
                relerr_daily=read_proton_daily_relerr(args.proton_meta,
                                                      rows=test_idx),
                relerr_parts=read_proton_relerr_parts(args.proton_meta,
                                                      rows=test_idx),
            )
            plot_paths = list(plot_paths) + diag_paths
        except Exception as exc:
            print(f"[skip test diagnostic plots] {exc}")

    summary = {
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "device": str(device),
        "neutron": str(args.neutron),
        "proton": str(args.proton),
        "aligned_days": int(n_days),
        "stations": used_stations,
        "n_stations": int(n_stations),
        "n_bins": int(n_bins),
        "kernels": kernels,
        "trainable_parameters": int(n_params),
        "data_seed": args.data_seed,
        "epochs": args.epochs,
        "epochs_run": len(train_losses),
        "best_epoch": best_epoch,
        # early-stopping criterion (MSE + L2 on the scaled target); 小导 picks
        # the data_seed with the lowest value -> --select-seed
        "best_val_loss": float(best_val) if best_epoch > 0 else None,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "l2": args.l2,
        "dropout": args.dropout,
        "init": args.init,
        "drop_last": args.drop_last,
        "test_frac": args.test_frac,
        "val_frac": args.val_frac,
        "early_stopping": {"patience": args.patience, "min_delta": args.min_delta},
        "target_transform": "log10" if y_is_log10 else "linear",
        "metrics": metrics,
        "outputs": {
            "model_dir": str(model_dir),
            "checkpoint": str(ckpt_path),
            "scalers": str(scaler_path),
            "scalers_csv": str(scaler_csv_path),
            "figure_dir": str(fig_dir),
            "error_dir": str(err_dir),
            "plots": plot_paths,
            "timeseries_dir": str(fig_dir / "timeseries"),
            "timeseries_plots": ts_paths,
        },
    }
    summary_path = args.output_dir / "n2p_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
