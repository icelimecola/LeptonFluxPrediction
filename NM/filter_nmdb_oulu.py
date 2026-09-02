#!/usr/bin/env python3
"""Filter daily NM count rates using OULU as the reference station."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from download_nmdb import DEFAULT_STATIONS, parse_stations
from station_metadata import CUTOFF_RIGIDITY_GV


REFERENCE_STATION = "OULU"
MOVING_AVERAGE_DAYS = 60
KDE_GRID_POINTS = 2000
KDE_LOWER_QUANTILE = 0.0015
KDE_UPPER_QUANTILE = 0.9985
PLOT_GROUP_SIZE = 6
MANUAL_RATIO_UPPER_LIMITS = {
    "SOPO": 3.0,
    "FSMT": 2.1,
}


@dataclass
class StationResult:
    station: str
    input_values: np.ndarray
    reference_values: np.ndarray
    raw_ratio: np.ndarray
    ratio_after_manual: np.ndarray
    moving_average: np.ndarray
    deviation: np.ndarray
    manual_outlier: np.ndarray
    kde_outlier: np.ndarray
    final_ratio: np.ndarray
    final_counts: np.ndarray
    kde_lower: float | None
    kde_upper: float | None
    kde_x: np.ndarray | None
    kde_pdf: np.ndarray | None


def parse_iso_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError(f"invalid date in input CSV: {value!r}") from exc


def parse_optional_float(value: str | None) -> float:
    if value is None or not value.strip():
        return np.nan
    try:
        number = float(value)
    except ValueError:
        return np.nan
    return number if math.isfinite(number) else np.nan


def read_station_csv(path: Path) -> tuple[list[dt.date], np.ndarray]:
    dates: list[dt.date] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"date", "daily_value_candidate"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(
                f"{path} must contain columns: {', '.join(sorted(required))}"
            )
        for row in reader:
            dates.append(parse_iso_date(row["date"]))
            values.append(parse_optional_float(row["daily_value_candidate"]))
    if not dates:
        raise RuntimeError(f"input CSV has no data rows: {path}")
    if any(current <= previous for previous, current in zip(dates, dates[1:])):
        raise RuntimeError(f"dates are not strictly increasing: {path}")
    return dates, np.asarray(values, dtype=np.float64)


def moving_average_nan_fixed(values: np.ndarray, window: int = 60) -> np.ndarray:
    """Reproduce the edge handling and window widths in the source script."""
    values = np.asarray(values, dtype=np.float64)
    size = values.size
    if window < 2:
        raise ValueError("moving-average window must be at least 2")
    if size < window:
        raise ValueError(
            f"time series has {size} rows, fewer than the {window}-day window"
        )

    half = window // 2
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    cumulative_sum = np.cumsum(np.insert(filled, 0, 0.0))
    cumulative_count = np.cumsum(np.insert(valid.astype(int), 0, 0))
    output = np.full(size, np.nan, dtype=np.float64)

    def interval_mean(left: int, right: int) -> float:
        total = cumulative_sum[right] - cumulative_sum[left]
        count = cumulative_count[right] - cumulative_count[left]
        return total / count if count > 0 else np.nan

    for index in range(half + 1):
        output[index] = interval_mean(index, index + window)
    for index in range(half + 1, size - half):
        output[index] = interval_mean(index - half, index + half + 1)
    for index in range(size - half, size):
        output[index] = interval_mean(index - window + 1, index + 1)
    return output


def kde_limits(
    values: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    try:
        from scipy.stats import gaussian_kde
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for OULU KDE filtering; install it with "
            "'python -m pip install scipy'"
        ) from exc

    finite = values[np.isfinite(values)]
    if finite.size < 2 or np.ptp(finite) == 0:
        raise RuntimeError("not enough varying deviations for KDE")
    kde = gaussian_kde(finite)
    grid = np.linspace(finite.min(), finite.max(), KDE_GRID_POINTS)
    density = kde(grid)
    spacing = grid[1] - grid[0]
    cdf = np.cumsum(density) * spacing
    if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
        raise RuntimeError("KDE integration did not produce a valid CDF")
    cdf /= cdf[-1]
    lower = float(np.interp(KDE_LOWER_QUANTILE, cdf, grid))
    upper = float(np.interp(KDE_UPPER_QUANTILE, cdf, grid))
    return lower, upper, grid, density


def filter_station(
    station: str,
    values: np.ndarray,
    reference_values: np.ndarray,
) -> StationResult:
    raw_ratio = np.full(values.shape, np.nan, dtype=np.float64)
    usable = (
        np.isfinite(values)
        & np.isfinite(reference_values)
        & (reference_values != 0)
    )
    raw_ratio[usable] = values[usable] / reference_values[usable]

    ratio_after_manual = raw_ratio.copy()
    manual_outlier = np.zeros(values.shape, dtype=bool)
    manual_upper = MANUAL_RATIO_UPPER_LIMITS.get(station)
    if manual_upper is not None:
        manual_outlier = np.isfinite(raw_ratio) & (raw_ratio > manual_upper)
        ratio_after_manual[manual_outlier] = np.nan

    moving_average = moving_average_nan_fixed(
        ratio_after_manual, MOVING_AVERAGE_DAYS
    )
    deviation = ratio_after_manual - moving_average
    kde_outlier = np.zeros(values.shape, dtype=bool)
    kde_lower: float | None = None
    kde_upper: float | None = None
    kde_x: np.ndarray | None = None
    kde_pdf: np.ndarray | None = None
    final_ratio = ratio_after_manual.copy()

    if station != REFERENCE_STATION:
        kde_lower, kde_upper, kde_x, kde_pdf = kde_limits(deviation)
        kde_outlier = np.isfinite(deviation) & (
            (deviation < kde_lower) | (deviation > kde_upper)
        )
        final_ratio[kde_outlier] = np.nan

    final_counts = final_ratio * reference_values
    return StationResult(
        station=station,
        input_values=values,
        reference_values=reference_values,
        raw_ratio=raw_ratio,
        ratio_after_manual=ratio_after_manual,
        moving_average=moving_average,
        deviation=deviation,
        manual_outlier=manual_outlier,
        kde_outlier=kde_outlier,
        final_ratio=final_ratio,
        final_counts=final_counts,
        kde_lower=kde_lower,
        kde_upper=kde_upper,
        kde_x=kde_x,
        kde_pdf=kde_pdf,
    )


def format_float(value: float) -> str:
    return f"{value:.12g}" if np.isfinite(value) else ""


def write_station_csv(
    path: Path, dates: list[dt.date], result: StationResult
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "date",
                "daily_value_candidate",
                "oulu_daily_value_candidate",
                "ratio_to_oulu_raw",
                "manual_ratio_outlier",
                "ratio_after_manual_filter",
                "ratio_moving_average_60d",
                "ratio_deviation",
                "kde_lower",
                "kde_upper",
                "kde_outlier",
                "oulu_filter_outlier",
                "final_ratio_to_oulu",
                "final_count_rate",
            ]
        )
        for index, day in enumerate(dates):
            writer.writerow(
                [
                    day.isoformat(),
                    format_float(result.input_values[index]),
                    format_float(result.reference_values[index]),
                    format_float(result.raw_ratio[index]),
                    int(result.manual_outlier[index]),
                    format_float(result.ratio_after_manual[index]),
                    format_float(result.moving_average[index]),
                    format_float(result.deviation[index]),
                    "" if result.kde_lower is None else f"{result.kde_lower:.12g}",
                    "" if result.kde_upper is None else f"{result.kde_upper:.12g}",
                    int(result.kde_outlier[index]),
                    int(
                        result.manual_outlier[index]
                        or result.kde_outlier[index]
                    ),
                    format_float(result.final_ratio[index]),
                    format_float(result.final_counts[index]),
                ]
            )


def write_combined_csv(
    path: Path,
    dates: list[dt.date],
    stations: list[str],
    results: dict[str, StationResult],
    attribute: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", *stations])
        arrays = [getattr(results[station], attribute) for station in stations]
        for index, day in enumerate(dates):
            writer.writerow(
                [day.isoformat(), *(format_float(array[index]) for array in arrays)]
            )


def station_label(station: str) -> str:
    return f"{station}  Rc={CUTOFF_RIGIDITY_GV[station]:g} GV"


def load_matplotlib(output_dir: Path):
    cache_dir = output_dir / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    return plt, mdates


def plot_ratio_panel(
    axis,
    dates,
    result: StationResult,
    mdates,
    *,
    combined: bool,
) -> None:
    removed = result.manual_outlier | result.kde_outlier
    axis.plot(
        dates,
        result.raw_ratio,
        linestyle="none",
        marker=".",
        markersize=1.7 if combined else 2.2,
        color="#5f6368",
        markeredgewidth=0,
        label="Ratio" if not combined else None,
    )
    axis.plot(
        dates,
        result.moving_average,
        color="#1a73e8",
        linewidth=1.0 if combined else 1.2,
        label="60-day mean" if not combined else None,
    )
    if removed.any():
        axis.scatter(
            np.asarray(dates)[removed],
            result.raw_ratio[removed],
            color="#c5221f",
            marker="x",
            s=11,
            linewidths=0.7,
            zorder=3,
            label="Filter candidate" if not combined else None,
        )
    axis.set_xlim(dates[0], dates[-1])
    axis.set_xlabel("Year")
    axis.set_ylabel("Count rate / OULU")
    axis.xaxis.set_major_locator(
        mdates.AutoDateLocator(
            minticks=4 if combined else 5,
            maxticks=7 if combined else 9,
        )
    )
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.tick_params(direction="in", top=True, right=True)
    axis.text(
        0.02,
        0.94,
        f"{result.station} {CUTOFF_RIGIDITY_GV[result.station]:.2f} GV",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11 if combined else 14,
        fontweight="bold",
    )
    if not combined:
        axis.legend(loc="best", fontsize=9, frameon=False)


def plot_deviation_distribution_panel(axis, result: StationResult) -> None:
    axis.text(
        0.02,
        0.88,
        station_label(result.station),
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )
    finite = result.deviation[np.isfinite(result.deviation)]
    if finite.size and np.ptp(finite) > 0:
        axis.hist(
            finite,
            bins=70,
            density=True,
            color="#9aa0a6",
            alpha=0.65,
            linewidth=0,
        )
    elif finite.size:
        axis.axvline(0.0, color="#5f6368", linewidth=1.0)
    if result.kde_x is not None and result.kde_pdf is not None:
        axis.plot(result.kde_x, result.kde_pdf, color="#1a73e8", linewidth=1.0)
    if result.kde_lower is not None and result.kde_upper is not None:
        axis.axvline(result.kde_lower, color="#c5221f", linewidth=0.9)
        axis.axvline(result.kde_upper, color="#c5221f", linewidth=0.9)
    if result.station == REFERENCE_STATION:
        axis.text(0.98, 0.88, "Reference", transform=axis.transAxes,
                  fontsize=8, ha="right", va="top")
    axis.set_xlabel("Ratio deviation from 60-day mean")
    axis.set_ylabel("Density")
    axis.grid(alpha=0.18, linewidth=0.5)


def plot_daily_counts_panel(
    axis,
    dates,
    result: StationResult,
    mdates,
    *,
    combined: bool,
) -> None:
    axis.plot(
        dates,
        result.final_counts,
        linestyle="none",
        marker=".",
        markersize=1.7 if combined else 2.2,
        color="#178314",
        markeredgewidth=0,
    )
    axis.set_xlim(dates[0], dates[-1])
    axis.set_xlabel("Year")
    axis.set_ylabel("Count rate (counts/s)")
    axis.xaxis.set_major_locator(
        mdates.AutoDateLocator(
            minticks=4 if combined else 5,
            maxticks=7 if combined else 9,
        )
    )
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.tick_params(direction="in", top=True, right=True)
    axis.text(
        0.02,
        0.94,
        f"{result.station} {CUTOFF_RIGIDITY_GV[result.station]:.2f} GV",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11 if combined else 14,
        fontweight="bold",
    )


def write_plots(
    plot_dir: Path,
    dates: list[dt.date],
    stations: list[str],
    results: dict[str, StationResult],
    *,
    diagnostic_only: bool,
) -> list[Path]:
    plt, mdates = load_matplotlib(plot_dir.parent)
    ratio_dir = plot_dir / "ratio"
    kde_dir = plot_dir / "kde"
    ratio_dir.mkdir(parents=True, exist_ok=True)
    kde_dir.mkdir(parents=True, exist_ok=True)
    daily_dir = plot_dir / "daily"
    if not diagnostic_only:
        daily_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for station in stations:
        result = results[station]
        figure, axis = plt.subplots(figsize=(10.0, 3.2))
        plot_ratio_panel(axis, dates, result, mdates, combined=False)
        figure.tight_layout()
        path = ratio_dir / f"{station}_oulu_ratio.pdf"
        figure.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        outputs.append(path)

        figure, axis = plt.subplots(figsize=(6.0, 4.0))
        plot_deviation_distribution_panel(axis, result)
        figure.tight_layout()
        path = kde_dir / f"{station}_oulu_kde.pdf"
        figure.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        outputs.append(path)

        if not diagnostic_only:
            figure, axis = plt.subplots(figsize=(10.0, 3.2))
            plot_daily_counts_panel(
                axis, dates, result, mdates, combined=False
            )
            figure.tight_layout()
            path = daily_dir / f"{station}_daily_oulu_filtered.pdf"
            figure.savefig(path, format="pdf", bbox_inches="tight")
            plt.close(figure)
            outputs.append(path)

    date_suffix = f"{dates[0]:%Y%m%d}_{dates[-1]:%Y%m%d}"
    for group_start in range(0, len(stations), PLOT_GROUP_SIZE):
        group = stations[group_start : group_start + PLOT_GROUP_SIZE]
        group_number = group_start // PLOT_GROUP_SIZE + 1

        figure, axes = plt.subplots(3, 2, figsize=(13.0, 10.0), squeeze=False)
        occupied: set[tuple[int, int]] = set()
        for index, station in enumerate(group):
            row = index % 3
            column = index // 3
            occupied.add((row, column))
            plot_ratio_panel(
                axes[row, column],
                dates,
                results[station],
                mdates,
                combined=True,
            )
        for row in range(3):
            for column in range(2):
                if (row, column) not in occupied:
                    axes[row, column].set_visible(False)
        figure.tight_layout()
        path = plot_dir / (
            f"combined_oulu_ratio_group_{group_number}_{date_suffix}.pdf"
        )
        figure.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        outputs.append(path)

        figure, axes = plt.subplots(3, 2, figsize=(13.0, 10.0), squeeze=False)
        for axis, station in zip(axes.flat, group):
            plot_deviation_distribution_panel(axis, results[station])
        for axis in list(axes.flat)[len(group) :]:
            axis.set_visible(False)
        figure.tight_layout()
        path = plot_dir / (
            f"combined_oulu_kde_group_{group_number}_{date_suffix}.pdf"
        )
        figure.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        outputs.append(path)

        if not diagnostic_only:
            figure, axes = plt.subplots(3, 2, figsize=(13.0, 10.0), squeeze=False)
            occupied: set[tuple[int, int]] = set()
            for index, station in enumerate(group):
                row = index % 3
                column = index // 3
                occupied.add((row, column))
                plot_daily_counts_panel(
                    axes[row, column],
                    dates,
                    results[station],
                    mdates,
                    combined=True,
                )
            for row in range(3):
                for column in range(2):
                    if (row, column) not in occupied:
                        axes[row, column].set_visible(False)
            figure.tight_layout()
            path = plot_dir / (
                f"combined_daily_oulu_filtered_group_{group_number}_{date_suffix}.pdf"
            )
            figure.savefig(path, format="pdf", bbox_inches="tight")
            plt.close(figure)
            outputs.append(path)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Filter daily NM count rates using OULU ratios and KDE."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=base / "rawdata" / "nmdb_daily_iqr" / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "rawdata" / "nmdb_oulu_filter",
    )
    parser.add_argument(
        "--stations",
        type=parse_stations,
        default=list(DEFAULT_STATIONS),
        help="comma-separated stations (default: the paper's 18 stations)",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="skip PDF generation"
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="write ratio/KDE diagnostics without writing filtered data",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    stations = list(dict.fromkeys(args.stations))
    missing_metadata = sorted(set(stations) - CUTOFF_RIGIDITY_GV.keys())
    if missing_metadata:
        raise SystemExit(
            "missing cutoff-rigidity metadata for: " + ", ".join(missing_metadata)
        )
    if not args.input_dir.is_dir():
        raise SystemExit(f"input directory does not exist: {args.input_dir}")

    reference_path = args.input_dir / f"{REFERENCE_STATION}_daily_iqr.csv"
    if not reference_path.is_file():
        raise SystemExit(f"reference-station CSV does not exist: {reference_path}")
    dates, reference_values = read_station_csv(reference_path)

    values_by_station: dict[str, np.ndarray] = {}
    for station in stations:
        path = args.input_dir / f"{station}_daily_iqr.csv"
        if not path.is_file():
            raise SystemExit(f"station CSV does not exist: {path}")
        station_dates, values = read_station_csv(path)
        if station_dates != dates:
            raise SystemExit(f"date axis differs from {REFERENCE_STATION}: {path}")
        values_by_station[station] = values

    args.output_dir.mkdir(parents=True, exist_ok=True)
    station_data_dir = args.output_dir / "data" / "stations"
    results: dict[str, StationResult] = {}
    reports: list[dict] = []
    for index, station in enumerate(stations, start=1):
        print(f"[{index}/{len(stations)}] filter {station} against {REFERENCE_STATION}")
        try:
            result = filter_station(
                station, values_by_station[station], reference_values
            )
        except RuntimeError as exc:
            raise SystemExit(f"{station}: {exc}") from exc
        results[station] = result
        output_path = station_data_dir / f"{station}_oulu_filter.csv"
        if not args.diagnostic_only:
            write_station_csv(output_path, dates, result)
        input_count = int(np.isfinite(result.input_values).sum())
        final_count = int(np.isfinite(result.final_counts).sum())
        manual_count = int(result.manual_outlier.sum())
        kde_count = int(result.kde_outlier.sum())
        print(
            f"  input={input_count}, manual_outliers={manual_count}, "
            f"kde_outliers={kde_count}, final={final_count}"
        )
        reports.append(
            {
                "station": station,
                "input_values": input_count,
                "manual_ratio_upper": MANUAL_RATIO_UPPER_LIMITS.get(station),
                "manual_outliers": manual_count,
                "kde_lower": result.kde_lower,
                "kde_upper": result.kde_upper,
                "kde_outliers": kde_count,
                "final_values": final_count,
                "new_missing_values": input_count - final_count,
                "output": "" if args.diagnostic_only else str(output_path),
            }
        )

    data_dir = args.output_dir / "data"
    counts_path = data_dir / "nm_daily_oulu_filtered_counts.csv"
    ratios_path = data_dir / "nm_daily_oulu_filtered_ratios.csv"
    if not args.diagnostic_only:
        write_combined_csv(
            counts_path, dates, stations, results, "final_counts"
        )
        write_combined_csv(
            ratios_path, dates, stations, results, "final_ratio"
        )

    plot_outputs: list[Path] = []
    if not args.no_plots:
        plot_outputs = write_plots(
            args.output_dir / "plots",
            dates,
            stations,
            results,
            diagnostic_only=args.diagnostic_only,
        )

    summary = {
        "date_range": [dates[0].isoformat(), dates[-1].isoformat()],
        "reference_station": REFERENCE_STATION,
        "input_column": "daily_value_candidate",
        "moving_average_days": MOVING_AVERAGE_DAYS,
        "kde_grid_points": KDE_GRID_POINTS,
        "kde_cdf_quantiles": [KDE_LOWER_QUANTILE, KDE_UPPER_QUANTILE],
        "manual_ratio_upper_limits": MANUAL_RATIO_UPPER_LIMITS,
        "stations": reports,
        "combined_counts": "" if args.diagnostic_only else str(counts_path),
        "combined_ratios": "" if args.diagnostic_only else str(ratios_path),
        "plots": [str(path) for path in plot_outputs],
        "notes": [
            "This stage only removes additional values; it does not restore IQR outliers.",
            "Diagnostic-only mode does not write filtered station or combined data.",
            "OULU is not KDE-filtered because it is the reference station.",
            "final_count_rate equals final_ratio_to_oulu multiplied by the OULU count rate.",
        ],
    }
    summary_path = args.output_dir / "oulu_filter_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if not args.diagnostic_only:
        print(f"Combined counts: {counts_path}")
        print(f"Combined ratios: {ratios_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
