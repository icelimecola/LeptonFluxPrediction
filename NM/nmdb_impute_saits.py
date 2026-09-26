#!/usr/bin/env python3
"""Impute missing neutron-monitor daily counts with SAITS.

Reads the OULU-filtered daily count-rate matrix produced by
``nmdb_filter_kde_oulu.py`` and fills every gap (genuine missing days plus the
days removed by the OULU ratio/KDE filter) using the SAITS model from PyPOTS,
mirroring the reference implementation:
    FluxPrediction/neutron/process_neutron12_impute.py

Flow:
  * MinMaxScaler normalization (fit on finite values);
  * optional MCAR-30% evaluation on a 70/15/15 split (prints SAITS MAE over all
    test windows, scored on the artificially masked cells only);
  * full-matrix imputation by averaging overlapping 365-day window predictions;
  * inverse transform back to counts/s.

Outputs under OUTPUT_DIR:
  * nm_daily_imputed_counts.csv     (date x station, complete, counts/s)
  * nm_daily_imputed_counts.npy     (T x D numpy matrix, counts/s)
  * plots/imputed/<STATION>_imputed.pdf (per-station series; imputed '+' blue)
  * model/<timestamp>/SAITS.pypots  (best-epoch model weights + tensorboard
                                      log, written by PyPOTS; skip with
                                      --no-save-model)

Early stopping: ``--patience N`` stops training once the validation loss has
not improved for N consecutive epochs (default: off, i.e. run all --epochs).
Whether or not it triggers, PyPOTS restores the best-validation epoch before
imputing, and that epoch is recorded as ``best_epoch`` in the summary.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nmdb_download import DEFAULT_STATIONS
from station_metadata import (
    CUTOFF_RIGIDITY_GV,
    stations_by_cutoff_rigidity,
)


@dataclass
class ImputeResult:
    dates: list[dt.date]
    stations: list[str]
    matrix: np.ndarray          # (T, D) imputed counts/s, fully finite
    original: np.ndarray        # (T, D) original counts/s with NaN == gap
    mae: float | None


def read_counts_matrix(path: Path) -> tuple[list[dt.date], list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        stations = header[1:]
        dates: list[dt.date] = []
        rows: list[list[float]] = []
        for row in reader:
            if not row:
                continue
            dates.append(dt.date.fromisoformat(row[0]))
            values = []
            for cell in row[1:]:
                text = cell.strip()
                values.append(float(text) if text else math.nan)
            rows.append(values)
    if not dates:
        raise RuntimeError(f"no data rows in {path}")
    matrix = np.asarray(rows, dtype=np.float64)
    return dates, stations, matrix


def write_imputed_csv(path: Path, dates, stations, matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", *stations])
        for index, day in enumerate(dates):
            writer.writerow(
                [day.isoformat(), *(f"{value:.12g}" for value in matrix[index])]
            )


def parse_station_list(text: str) -> list[str]:
    """Parse a comma-separated station list, e.g. 'INVK,APTY,THUL'."""
    return [item.strip().upper() for item in text.split(",") if item.strip()]


def select_station_subset(
    data: np.ndarray,
    stations: list[str],
    requested: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    """Keep only the requested stations, preserving the input column order."""
    if not requested:
        return data, list(stations)
    order = {name: index for index, name in enumerate(stations)}
    missing = [name for name in requested if name not in order]
    if missing:
        raise SystemExit(
            "--stations not present in the input matrix: " + ", ".join(missing)
        )
    chosen = sorted(dict.fromkeys(requested), key=lambda name: order[name])
    indices = [order[name] for name in chosen]
    return data[:, indices], chosen


def apply_mcar_mask(matrix: np.ndarray, rate: float, seed: int) -> np.ndarray:
    """Mask ``rate`` of the observed values, leaving original NaN untouched."""
    rng = np.random.default_rng(seed)
    masked = matrix.copy()
    t_idx, d_idx = np.where(np.isfinite(matrix))
    n_mask = int(rate * t_idx.size)
    chosen = rng.choice(t_idx.size, size=n_mask, replace=False)
    masked[t_idx[chosen], d_idx[chosen]] = np.nan
    return masked


def sliding_windows(matrix: np.ndarray, window: int) -> np.ndarray:
    """Overlapping windows (n_windows, window, D) with stride 1."""
    if matrix.shape[0] < window:
        raise ValueError(
            f"series has {matrix.shape[0]} rows, fewer than window {window}"
        )
    return np.stack(
        [matrix[i : i + window, :] for i in range(matrix.shape[0] - window + 1)],
        axis=0,
    )


def _load_pypots():
    try:
        from pypots.imputation import SAITS
        from pypots.nn.functional import calc_mae
    except ImportError as exc:
        raise RuntimeError(
            "nmdb_impute_saits.py requires PyPOTS (and its deps torch/sklearn); "
            "install it in the run environment then rerun"
        ) from exc
    return SAITS, calc_mae


def make_model(
    window: int,
    n_features: int,
    epochs: int,
    patience: int | None = None,
    saving_path: Path | None = None,
):
    SAITS, _ = _load_pypots()
    return SAITS(
        n_steps=window,
        n_features=n_features,
        n_layers=2,
        d_model=256,
        n_heads=4,
        d_k=64,
        d_v=64,
        d_ffn=128,
        dropout=0.1,
        epochs=epochs,
        # early stopping: None keeps the old behaviour (run every epoch)
        patience=patience,
        # PyPOTS writes the best model (+ a tensorboard log) under this
        # directory in a timestamped sub-folder; None disables saving
        saving_path=str(saving_path) if saving_path is not None else None,
        model_saving_strategy="best",
    )


class _LossLoggerHandler(logging.Handler):
    """Capture per-epoch train/val loss from PyPOTS training logs.

    PyPOTS does not expose a per-epoch loss history after ``fit``; it only logs
    lines like ``Epoch 003 - training loss (MAE): 0.0521, validation MSE: 0.0031``.
    This handler parses those records into ``(epoch, train_loss, val_loss)``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[tuple[int, float, float | None]] = []
        # epoch PyPOTS restored at the end ("The best model is from epoch#N")
        self.best_epoch: int | None = None

    def emit(self, record: logging.LogRecord) -> None:
        best = re.search(r"best model is from epoch#(\d+)", record.getMessage())
        if best is not None:
            self.best_epoch = int(best.group(1))
            return
        if "Epoch" not in record.getMessage() or "training loss" not in record.getMessage():
            return
        text = record.getMessage()
        m = re.search(r"Epoch (\d+) - training loss \(.*?\): ([\d.eE+-]+)", text)
        if m is None:
            return
        epoch = int(m.group(1))
        train_loss = float(m.group(2))
        v = re.search(r"validation .*?: ([\d.eE+-]+)", text)
        val_loss = float(v.group(1)) if v is not None else None
        self.records.append((epoch, train_loss, val_loss))


def plot_loss_history(
    plot_dir: Path,
    history: list[tuple[int, float, float | None]],
    test_mae: float | None,
) -> Path:
    """Plot training/validation loss vs epoch, styled like the LSTM figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / "model_loss.pdf"

    epochs_list = [item[0] for item in history]
    train_loss = [item[1] for item in history]
    val_epochs = [item[0] for item in history if item[2] is not None]
    val_loss = [item[2] for item in history if item[2] is not None]

    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    axis.plot(epochs_list, train_loss, label="training (MAE)")          # 默认色（蓝）
    if val_loss:
        axis.plot(val_epochs, val_loss, label="validation (MSE)")       # 默认色（橙）
    if test_mae is not None:
        axis.axhline(
            test_mae,
            color="C2",                                                  # 默认绿
            linestyle="--",
            linewidth=1.0,
            label=f"test (MAE) ({test_mae:.4f})",
        )
    axis.set_yscale("log")
    axis.set_title("model loss")
    axis.set_ylabel("loss")
    axis.set_xlabel("epoch")
    axis.tick_params(direction="in", top=True, right=True)
    axis.legend(loc="upper right", fontsize=9, frameon=False)
    figure.tight_layout()
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return path


def train_and_evaluate(
    scaled: np.ndarray,
    window: int,
    epochs: int,
    mask_rate: float,
    seed: int,
    n_features: int,
    do_eval: bool,
    patience: int | None = None,
    saving_path: Path | None = None,
) -> tuple[
    object,
    float | None,
    list[tuple[int, float, float | None]],
    dict[str, int] | None,
    int | None,
]:
    SAITS, calc_mae = _load_pypots()
    from pypots.utils.logging import logger as pypots_logger

    handler = _LossLoggerHandler()
    pypots_logger.addHandler(handler)

    t_steps = scaled.shape[0]

    if do_eval:
        n_train = int(0.70 * t_steps)
        n_val = int(0.15 * t_steps)
        train_ori = scaled[:n_train]
        val_ori = scaled[n_train : n_train + n_val]
        test_ori = scaled[n_train + n_val :]

        train_masked = apply_mcar_mask(train_ori, mask_rate, seed)
        val_masked = apply_mcar_mask(val_ori, mask_rate, seed)
        test_masked = apply_mcar_mask(test_ori, mask_rate, seed)

        train_w = sliding_windows(train_masked, window)
        val_w = sliding_windows(val_masked, window)
        val_ori_w = sliding_windows(val_ori, window)
        test_w = sliding_windows(test_masked, window)
        test_ori_w = sliding_windows(test_ori, window)

        model = make_model(window, n_features, epochs, patience, saving_path)
        try:
            model.fit({"X": train_w}, {"X": val_w, "X_ori": val_ori_w})
        finally:
            pypots_logger.removeHandler(handler)

        # Test MAE uses *all* test windows at once, matching the reference
        # implementation (FluxPrediction/neutron/process_neutron12_impute.py).
        # Only artificially masked cells are scored; cells that were already
        # missing have no ground truth and drop out of the XOR mask.
        # Because the windows overlap, a cell in the middle of the split is
        # counted `window` times while edge cells are counted less often, so
        # this is a window-weighted mean rather than a per-cell mean.
        mask = np.isnan(test_w) ^ np.isnan(test_ori_w)  # (N, window, D)
        imputed = model.impute({"X": test_w})           # (N, window, D)
        mae = float(calc_mae(imputed, np.nan_to_num(test_ori_w), mask))
        return model, mae, handler.records, {
            "train_end": n_train,
            "val_end": n_train + n_val,
        }, handler.best_epoch

    # No evaluation: train on the full-window set so the model can impute later.
    # No ground truth exists here, so pass no validation set.
    windows = sliding_windows(scaled, window)
    model = make_model(window, n_features, epochs, patience, saving_path)
    try:
        model.fit({"X": windows})
    finally:
        pypots_logger.removeHandler(handler)
    return model, None, handler.records, None, handler.best_epoch


def impute_full(scaled: np.ndarray, window: int, model) -> np.ndarray:
    """Impute every window of the full matrix and average the overlaps."""
    _, n_features = scaled.shape
    t_steps = scaled.shape[0]
    windows = sliding_windows(scaled, window)
    n_windows = windows.shape[0]

    imputed_full = np.zeros((t_steps, n_features))
    count = np.zeros((t_steps, n_features))
    for i in range(n_windows):
        pred = model.impute({"X": windows[i : i + 1]})  # (1, window, D)
        imputed_full[i : i + window, :] += pred[0]
        count[i : i + window, :] += 1.0
    imputed_full /= np.maximum(count, 1.0)
    return imputed_full


def plot_imputed(
    plot_dir: Path,
    dates,
    station: str,
    station_index: int,
    original: np.ndarray,
    imputed: np.ndarray,
    split_dates=None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    values = imputed[:, station_index]
    original_col = original[:, station_index]
    observed = np.isfinite(original_col)

    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"{station}_imputed.pdf"

    figure, axis = plt.subplots(figsize=(11.0, 3.4))
    axis.plot(
        dates, values,
        linestyle="none", marker=".", markersize=1.8,
        color="#5f6368", markeredgewidth=0, label="Observed",
    )
    imputed_index = np.where(~observed)[0]
    if imputed_index.size:
        axis.plot(
            [dates[i] for i in imputed_index], values[imputed_index],
            linestyle="none", marker="+", markersize=5.0,
            color="#1a73e8", markeredgewidth=0.8, label="Imputed",
        )
    axis.set_xlim(dates[0], dates[-1])
    if split_dates:
        for boundary, label in zip(split_dates, ("train|val", "val|test")):
            axis.axvline(
                boundary, color="0.3", linestyle=":", linewidth=1.0,
                alpha=0.7, zorder=0, label=label,
            )
    axis.set_xlabel("Year")
    axis.set_ylabel("Count rate (counts/s)")
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.tick_params(direction="in", top=True, right=True)
    axis.set_title(f"{station} — SAITS imputation", fontsize=13, fontweight="bold")
    axis.legend(loc="best", fontsize=9, frameon=False)
    figure.tight_layout()
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return path


PLOT_GROUP_SIZE = 6


def draw_imputed_panel(
    axis,
    dates,
    station: str,
    station_index: int,
    original: np.ndarray,
    imputed: np.ndarray,
    mdates,
    *,
    combined: bool,
    split_dates=None,
) -> None:
    """Draw one panel of the imputed daily series (gray observed + blue '')."""
    values = imputed[:, station_index]
    observed = np.isfinite(original[:, station_index])
    axis.plot(
        dates,
        values,
        linestyle="none",
        marker=".",
        markersize=1.7 if combined else 2.2,
        color="#5f6368",
        markeredgewidth=0,
        label="Observed" if not combined else None,
    )
    imputed_index = np.where(~observed)[0]
    if imputed_index.size:
        axis.plot(
            [dates[i] for i in imputed_index],
            values[imputed_index],
            linestyle="none",
            marker="+",
            markersize=5.0 if combined else 6.0,
            color="#1a73e8",
            markeredgewidth=0.8,
            label="Imputed" if not combined else None,
        )
    axis.set_xlim(dates[0], dates[-1])
    if split_dates:
        for boundary in split_dates:
            axis.axvline(
                boundary, color="0.3", linestyle=":", linewidth=1.0,
                alpha=0.7, zorder=0,
            )
    axis.set_xlabel("Year")
    axis.set_ylabel("Count rate (counts/s)")
    axis.xaxis.set_major_locator(
        mdates.AutoDateLocator(
            minticks=4 if combined else 5, maxticks=7 if combined else 9
        )
    )
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.tick_params(direction="in", top=True, right=True)
    axis.text(
        0.02,
        0.94,
        f"{station} {CUTOFF_RIGIDITY_GV[station]:.2f} GV",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11 if combined else 14,
        fontweight="bold",
    )
    if not combined:
        axis.legend(loc="best", fontsize=9, frameon=False)


def write_combined_imputed(
    plot_dir: Path,
    dates,
    stations,
    original: np.ndarray,
    imputed: np.ndarray,
    split_dates=None,
) -> list[str]:
    """Write combined 3x2 group PDFs of the imputed series (6 stations/group)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    date_suffix = f"{dates[0]:%Y%m%d}_{dates[-1]:%Y%m%d}"
    for group_start in range(0, len(stations), PLOT_GROUP_SIZE):
        group = stations[group_start : group_start + PLOT_GROUP_SIZE]
        group_number = group_start // PLOT_GROUP_SIZE + 1
        figure, axes = plt.subplots(3, 2, figsize=(13.0, 10.0), squeeze=False)
        for index, station in enumerate(group):
            row = index // 2
            column = index % 2
            station_index = stations.index(station)
            draw_imputed_panel(
                axes[row, column],
                dates,
                station,
                station_index,
                original,
                imputed,
                mdates,
                combined=True,
                split_dates=split_dates,
            )
        for row in range(3):
            for column in range(2):
                if (row, column) not in {
                    (index // 2, index % 2) for index in range(len(group))
                }:
                    axes[row, column].set_visible(False)
        figure.suptitle(
            f"SAITS imputation (group {group_number})",
            fontsize=15,
            fontweight="bold",
        )
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        path = plot_dir / (
            f"combined_imputed_group_{group_number}_{date_suffix}.pdf"
        )
        figure.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        outputs.append(str(path))
    return outputs


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Impute missing NM daily counts with SAITS."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            base / "data" / "nmdb_filter_kde_oulu" / "data"
            / "nm_daily_oulu_filtered_counts.npy"
        ),
        help=(
            "input matrix: .npy (T,D) with NaN gaps (default), .npz "
            "('counts' member), or the legacy .csv (date x station)"
        ),
    )
    parser.add_argument(
        "--start-date",
        default="2001-01-01",
        help=(
            "first date of the consecutive daily grid used to label rows when "
            "reading a bare .npy/.npz matrix, which carries no dates of its own. "
            "The KDE matrix now starts at 2001-01-01; a stale value here does not "
            "fail loudly, it shifts every output date by the difference, so keep "
            "it in step with the range the chain was run over (default: 2001-01-01)"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=base / "data" / "nmdb_imputed",
    )
    parser.add_argument("--window", type=int, default=365)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mask-rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help=(
            "early stopping: stop after this many epochs without a better "
            "validation loss (default: off, run all --epochs). With "
            "--skip-eval there is no validation set, so PyPOTS monitors the "
            "training loss instead"
        ),
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="do not save the trained model under OUTPUT_DIR/model",
    )
    parser.add_argument(
        "--skip-eval", action="store_true", help="skip the MCAR-30%% MAE check"
    )
    parser.add_argument("--no-plots", action="store_true", help="skip PDFs")
    parser.add_argument(
        "--stations",
        type=parse_station_list,
        default=None,
        help=(
            "comma-separated subset of stations to impute, e.g. "
            "'INVK,APTY,THUL' (default: all stations in the input matrix). "
            "The subset keeps the column order of the input matrix."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.window < 2:
        raise SystemExit("--window must be at least 2")
    if not 0 < args.mask_rate < 1:
        raise SystemExit("--mask-rate must be between 0 and 1")
    if args.patience is not None and args.patience < 1:
        raise SystemExit("--patience must be a positive integer")
    if not args.input.is_file():
        raise SystemExit(f"input file does not exist: {args.input}")

    if args.input.suffix == ".csv":
        # 兼容旧输入：CSV 自带日期与站名
        dates, stations, data = read_counts_matrix(args.input)
    else:
        if args.input.suffix == ".npz":
            with np.load(args.input) as npz_file:
                data = np.array(npz_file["counts"])
                stations = (
                    [str(name) for name in npz_file["stations"]]
                    if "stations" in npz_file.files
                    else stations_by_cutoff_rigidity(list(DEFAULT_STATIONS))
                )
        else:
            data = np.load(args.input)
            stations = stations_by_cutoff_rigidity(list(DEFAULT_STATIONS))
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise SystemExit("input matrix must be 2-D (T, D)")
        # 纯数值矩阵：日期按连续日网格约定重建
        try:
            start_date = dt.date.fromisoformat(args.start_date)
        except ValueError as exc:
            raise SystemExit(f"--start-date invalid: {args.start_date}") from exc
        dates = [start_date + dt.timedelta(days=i) for i in range(data.shape[0])]
    if data.shape[1] != len(stations):
        raise SystemExit("matrix columns do not match station set")

    # optional station subset (e.g. only the stations similar to OULU)
    data, stations = select_station_subset(data, stations, args.stations)

    if len(dates) != data.shape[0]:
        raise SystemExit("date axis length does not match matrix rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    model_dir = None if args.no_save_model else args.output_dir / "model"
    model, mae, loss_history, split_index, best_epoch = train_and_evaluate(
        scaled,
        args.window,
        args.epochs,
        args.mask_rate,
        args.seed,
        data.shape[1],
        do_eval=not args.skip_eval,
        patience=args.patience,
        saving_path=model_dir,
    )
    epochs_run = loss_history[-1][0] if loss_history else None
    if best_epoch is not None:
        stopped = (
            f", stopped early at epoch {epochs_run}"
            if epochs_run is not None and epochs_run < args.epochs
            else ""
        )
        print(f"best epoch (restored before imputing): {best_epoch}{stopped}")
    saved_models = (
        sorted(str(path) for path in model_dir.rglob("*.pypots"))
        if model_dir is not None and model_dir.is_dir()
        else []
    )
    if model_dir is not None:
        print(f"saved model: {saved_models[-1] if saved_models else '(none found)'}")
    # vertical split lines for the per-station plots (train|val, val|test)
    split_dates = None
    if split_index:
        split_dates = (
            dates[split_index["train_end"]],
            dates[split_index["val_end"]],
        )
        print(
            f"split boundaries: train|val {split_dates[0]}, "
            f"val|test {split_dates[1]}"
        )
    if mae is not None:
        print(
            f"SAITS MAE (MCAR {args.mask_rate:.0%}, all test windows): {mae:.6f}"
        )
    if loss_history:
        first_epoch = loss_history[0][0]
        last_epoch = loss_history[-1][0]
        train_last = loss_history[-1][1]
        print(
            f"train loss: epoch {first_epoch} -> {last_epoch}, "
            f"final train loss {train_last:.6f}"
        )

    print("Imputing full matrix ...")
    imputed_scaled = impute_full(scaled, args.window, model)
    imputed = scaler.inverse_transform(imputed_scaled)

    csv_path = args.output_dir / "nm_daily_imputed_counts.csv"
    write_imputed_csv(csv_path, dates, stations, imputed)
    npy_path = args.output_dir / "nm_daily_imputed_counts.npy"
    np.save(npy_path, imputed)

    output_plots: list[str] = []
    if not args.no_plots:
        plots_root = args.output_dir / "plots"
        imputed_dir = plots_root / "imputed"
        for index, station in enumerate(stations):
            output_plots.append(
                str(plot_imputed(
                    imputed_dir, dates, station, index, data, imputed,
                    split_dates=split_dates,
                ))
            )
        output_plots.extend(
            write_combined_imputed(
                plots_root, dates, stations, data, imputed,
                split_dates=split_dates,
            )
        )
        if loss_history:
            output_plots.append(
                str(plot_loss_history(plots_root, loss_history, mae))
            )

    summary = {
        "date_range": [dates[0].isoformat(), dates[-1].isoformat()],
        "n_days": len(dates),
        "stations": stations,
        "n_stations": len(stations),
        "requested_stations": args.stations,
        "split_day_counts": split_index,
        "split_boundary_dates": (
            [day.isoformat() for day in split_dates] if split_dates else None
        ),
        "window": args.window,
        "epochs": args.epochs,
        "patience": args.patience,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "saved_model": saved_models[-1] if saved_models else None,
        "mask_rate": args.mask_rate,
        "seed": args.seed,
        "mae": mae,
        "csv": str(csv_path),
        "npy": str(npy_path),
        "plots": output_plots,
        "notes": [
            "all gaps were imputed, including days removed by the OULU filter",
            "final matrix is complete (no NaN)",
        ],
    }
    summary_path = args.output_dir / "impute_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"CSV: {csv_path}")
    print(f"npy: {npy_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
