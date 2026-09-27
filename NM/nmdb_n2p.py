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
random 80/10/10 split controlled by --data-seed, Adamax, MSE loss,
early stopping on val_loss + keep the best-val weights.
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
def read_neutron(path: Path, start_date: dt.date):
    data = np.load(path)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise SystemExit(f"neutron matrix must be 2-D, got {data.shape}")
    dates = [start_date + dt.timedelta(days=i) for i in range(data.shape[0])]
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


def adaptive_kernels(n_stations: int) -> list[int]:
    """Kernel sizes clipped to the number of stations (mirrors 7/5/3)."""
    kernels = sorted({min(7, n_stations), min(5, n_stations), min(3, n_stations)})
    return list(reversed(kernels))


# --------------------------------------------------------------------------
# model (PyTorch)
# --------------------------------------------------------------------------
def build_model(n_stations: int, n_bins: int, dropout: float):
    import torch
    import torch.nn as nn

    kernels = adaptive_kernels(n_stations)

    def conv_stack():
        layers = []
        for k in kernels:
            layers += [
                nn.Conv1d(64, 64, k, padding="same"),
                nn.BatchNorm1d(64),
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
                    nn.BatchNorm1d(64),
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
    return model, kernels


def inverse_log_minmax(values: np.ndarray, scaler) -> np.ndarray:
    """scaled -> log10(flux) -> flux."""
    return np.power(10.0, scaler.inverse_transform(values))


def predict_flux(model, X, scaler_y, device):
    import torch

    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        pred_scaled = model(tensor).cpu().numpy()
    return inverse_log_minmax(pred_scaled, scaler_y)


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
    axis.set_xlabel("observed flux")
    axis.set_ylabel("predicted flux")
    axis.set_title("test: predicted vs observed")
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
    axis.set_xlabel("rigidity bin index")
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
            axis.set_xlabel("observed", fontsize=9)
            axis.set_ylabel("predicted", fontsize=9)
            axis.set_title(f"bin {j + 1}: {bins[j][0]:g}-{bins[j][1]:g} GV",
                           fontsize=11, fontweight="bold")
            axis.tick_params(direction="in", top=True, right=True, labelsize=8)
        for index in range(len(group), 6):
            axes[index // 2, index % 2].set_visible(False)
        figure.suptitle(
            "test: predicted vs observed (log-log)",
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

    colors = {"train": "C0", "val": "C1", "test": "C3"}
    indiv_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    date_lo, date_hi = dates[0], dates[-1]

    def style_axis(axis, combined):
        axis.set_yscale("log")
        axis.set_xlim(date_lo, date_hi)
        axis.set_ylabel("flux (m$^{-2}$sr$^{-1}$s$^{-1}$GV$^{-1}$)",
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
            0.02, 0.94, f"bin {j + 1}: {bins[j][0]:g}-{bins[j][1]:g} GV",
            transform=axis.transAxes, ha="left", va="top",
            fontsize=11 if combined else 13, fontweight="bold",
        )

    # individual: detailed, one bin per figure, predicted split by colour
    for j in range(obs.shape[1]):
        figure, axis = plt.subplots(figsize=(11.0, 3.4))
        axis.plot(dates, obs[:, j], linestyle="none", marker=".",
                  markersize=2.6, color="0.6", markeredgewidth=0, zorder=1,
                  label="observed")
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
            loc="upper left", bbox_to_anchor=(0.0, 0.86),
            fontsize=9, frameon=False, markerscale=1.8, handlelength=1.2,
            labelcolor=["0.3", "C0", "C1", "C3"],
        )
        figure.tight_layout()
        path = indiv_dir / f"bin_{j + 1:02d}_{bins[j][0]:g}-{bins[j][1]:g}GV.pdf"
        figure.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        paths.append(str(path))

    # combined: simple and sparse, 6 bins per 3x2 figure, observed grey + predicted blue
    n_bins = obs.shape[1]
    for group_start in range(0, n_bins, group_size):
        group = list(range(group_start, min(group_start + group_size, n_bins)))
        group_number = group_start // group_size + 1
        figure, axes = plt.subplots(3, 2, figsize=(16.0, 11.0), squeeze=False)
        for index, j in enumerate(group):
            axis = axes[index // 2, index % 2]
            axis.plot(dates, obs[:, j], linestyle="none", marker=".",
                      markersize=2.6, color="0.55", markeredgewidth=0, zorder=1,
                      label="observed")
            axis.plot(dates, prd[:, j], linestyle="none", marker=".",
                      markersize=2.4, color="#1a73e8", markeredgewidth=0, zorder=2,
                      label="predicted")
            style_axis(axis, combined=True)
            bin_label(axis, j, combined=True)
            axis.legend(
                loc="upper left", bbox_to_anchor=(0.0, 0.86),
                fontsize=9, frameon=False, markerscale=2.2, handlelength=1.2,
                labelcolor=["0.3", "#1a73e8"],
            )
        for index in range(len(group), group_size):
            axes[index // 2, index % 2].set_visible(False)
        figure.suptitle(
            "proton flux: observed (grey) vs predicted (blue) "
            f"(bins {group[0] + 1}-{group[-1] + 1})",
            fontsize=15, fontweight="bold",
        )
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        path = combined_dir / f"combined_timeseries_group_{group_number}.pdf"
        figure.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        paths.append(str(path))
    return paths


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
        "--neutron-start", default="2011-01-01",
        help="date of the first row of the neutron matrix",
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
        "--device", default=None,
        help="torch device (default: cuda > mps > cpu)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=base / "data" / "n2p",
        help="root for Model/ Figure/ Error/ outputs (default: NM/data/n2p)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="fast smoke run: epochs<=500 and write into <output-dir>/quick",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=42, help="global seed for weight init etc."
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

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

    start_date = dt.date.fromisoformat(args.neutron_start)
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

    y_log = np.log10(y_raw)

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

    model, kernels = build_model(n_stations, n_bins, args.dropout)
    model.to(device)
    print("kernels:", kernels)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params:,}")

    # L2 only on the output layer (mirrors L2 on the Keras Dense kernel)
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    other_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.Adamax(
        [
            {"params": other_params, "weight_decay": 0.0},
            {"params": head_params, "weight_decay": args.l2},
        ],
        lr=args.lr, betas=(0.9, 0.999), eps=1e-7,
    )
    criterion = torch.nn.MSELoss()

    X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train_s, dtype=torch.float32)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val_s, dtype=torch.float32, device=device)
    X_test_t = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.as_tensor(y_test_s, dtype=torch.float32, device=device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_t, y_train_t),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )

    model_dir = args.output_dir / "Model" / "n2p"
    fig_dir = args.output_dir / "Figure" / "n2p"
    err_dir = args.output_dir / "Error" / "n2p"
    for path in (model_dir, fig_dir, err_dir):
        path.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seed{args.data_seed}_{n_stations}st_"
        f"{args.epochs}ep_lr{args.lr}_l2{args.l2}_do{args.dropout}_b{args.batch_size}"
    )
    ckpt_path = model_dir / f"{tag}_best.pt"

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
        true_flux = np.power(10.0, y_log_true)
        pred_flux = predict_flux(model, X_in, scaler_y, device)
        true_by_split[name] = true_flux
        pred_by_split[name] = pred_flux
        np.save(err_dir / f"{tag}_{name}_true_flux.npy", true_flux)
        np.save(err_dir / f"{tag}_{name}_pred_flux.npy", pred_flux)
        metrics[name] = {
            "mse_log10": float(np.mean((np.log10(pred_flux) - y_log_true) ** 2)),
            "mae_log10": float(np.mean(np.abs(np.log10(pred_flux) - y_log_true))),
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
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "l2": args.l2,
        "dropout": args.dropout,
        "test_frac": args.test_frac,
        "val_frac": args.val_frac,
        "early_stopping": {"patience": args.patience, "min_delta": args.min_delta},
        "metrics": metrics,
        "outputs": {
            "model_dir": str(model_dir),
            "checkpoint": str(ckpt_path),
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
