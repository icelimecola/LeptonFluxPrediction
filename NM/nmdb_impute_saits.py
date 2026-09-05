#!/usr/bin/env python3
"""Impute missing neutron-monitor daily counts with SAITS.

Reads the OULU-filtered daily count-rate matrix produced by
``nmdb_filter_kde_oulu.py`` and fills every gap (genuine missing days plus the
days removed by the OULU ratio/KDE filter) using the SAITS model from PyPOTS,
mirroring the reference implementation:
    FluxPrediction/neutron/process_neutron12_impute.py

Flow:
  * MinMaxScaler normalization (fit on finite values);
  * optional MCAR-30% evaluation on a 70/15/15 split (prints SAITS MAE);
  * full-matrix imputation by averaging overlapping 365-day window predictions;
  * inverse transform back to counts/s.

Outputs under OUTPUT_DIR:
  * nm_daily_imputed_counts.csv     (date x station, complete, counts/s)
  * nm_daily_imputed_counts.npy     (T x D numpy matrix, counts/s)
  * plots/imputed/<STATION>_imputed.pdf (per-station series; imputed '+' blue)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
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


def make_model(window: int, n_features: int, epochs: int):
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
    )


def train_and_evaluate(
    scaled: np.ndarray,
    window: int,
    epochs: int,
    mask_rate: float,
    seed: int,
    n_features: int,
    do_eval: bool,
) -> tuple[object, float | None]:
    SAITS, calc_mae = _load_pypots()
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

        model = make_model(window, n_features, epochs)
        model.fit({"X": train_w}, {"X": val_w, "X_ori": val_ori_w})

        test_input = test_w[0][np.newaxis, ...]       # (1, window, D)
        test_ground = test_ori_w[0][np.newaxis, ...]
        imputed = model.impute({"X": test_input})
        mask = np.isnan(test_input) ^ np.isnan(test_ground)
        mae = float(calc_mae(imputed, np.nan_to_num(test_ground), mask))
        return model, mae

    # No evaluation: train on the full-window set so the model can impute later.
    # No ground truth exists here, so pass no validation set.
    windows = sliding_windows(scaled, window)
    model = make_model(window, n_features, epochs)
    model.fit({"X": windows})
    return model, None


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
            base / "rawdata" / "nmdb_filter_kde_oulu" / "data"
            / "nm_daily_oulu_filtered_counts.npy"
        ),
        help=(
            "input matrix: .npy (T,D) with NaN gaps (default), .npz "
            "('counts' member), or the legacy .csv (date x station)"
        ),
    )
    parser.add_argument(
        "--start-date",
        default="2011-01-01",
        help=(
            "first date of the consecutive daily grid used to label rows "
            "when reading a bare .npy/.npz matrix (default: 2011-01-01)"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=base / "rawdata" / "nmdb_imputed",
    )
    parser.add_argument("--window", type=int, default=365)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mask-rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--skip-eval", action="store_true", help="skip the MCAR-30%% MAE check"
    )
    parser.add_argument("--no-plots", action="store_true", help="skip PDFs")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.window < 2:
        raise SystemExit("--window must be at least 2")
    if not 0 < args.mask_rate < 1:
        raise SystemExit("--mask-rate must be between 0 and 1")
    if not args.input.is_file():
        raise SystemExit(f"input file does not exist: {args.input}")

    if args.input.suffix == ".csv":
        # 兼容旧输入：CSV 自带日期与站名
        dates, stations, data = read_counts_matrix(args.input)
    else:
        if args.input.suffix == ".npz":
            with np.load(args.input) as npz_file:
                data = np.array(npz_file["counts"])
        else:
            data = np.load(args.input)
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise SystemExit("input matrix must be 2-D (T, D)")
        # 纯数值矩阵：站名取默认 18 站（Rc 升序），日期按连续日网格约定重建
        stations = stations_by_cutoff_rigidity(list(DEFAULT_STATIONS))
        try:
            start_date = dt.date.fromisoformat(args.start_date)
        except ValueError as exc:
            raise SystemExit(f"--start-date invalid: {args.start_date}") from exc
        dates = [start_date + dt.timedelta(days=i) for i in range(data.shape[0])]
    if data.shape[1] != len(stations):
        raise SystemExit("matrix columns do not match station set")
    if len(dates) != data.shape[0]:
        raise SystemExit("date axis length does not match matrix rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    model, mae = train_and_evaluate(
        scaled,
        args.window,
        args.epochs,
        args.mask_rate,
        args.seed,
        data.shape[1],
        do_eval=not args.skip_eval,
    )
    if mae is not None:
        print(f"SAITS MAE (MCAR {args.mask_rate:.0%}): {mae:.6f}")

    print("Imputing full matrix ...")
    imputed_scaled = impute_full(scaled, args.window, model)
    imputed = scaler.inverse_transform(imputed_scaled)

    csv_path = args.output_dir / "nm_daily_imputed_counts.csv"
    write_imputed_csv(csv_path, dates, stations, imputed)
    npy_path = args.output_dir / "nm_daily_imputed_counts.npy"
    np.save(npy_path, imputed)

    output_plots: list[str] = []
    if not args.no_plots:
        plot_dir = args.output_dir / "plots" / "imputed"
        for index, station in enumerate(stations):
            output_plots.append(
                str(plot_imputed(plot_dir, dates, station, index, data, imputed))
            )
        output_plots.extend(
            write_combined_imputed(
                plot_dir, dates, stations, data, imputed
            )
        )

    summary = {
        "date_range": [dates[0].isoformat(), dates[-1].isoformat()],
        "n_days": len(dates),
        "stations": stations,
        "n_stations": len(stations),
        "window": args.window,
        "epochs": args.epochs,
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
