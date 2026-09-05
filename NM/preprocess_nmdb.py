#!/usr/bin/env python3
"""Apply two-stage 3*IQR filtering and compute daily NM count rates."""

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

from audit_nmdb import NULL_VALUES, parse_filename, scan_data_file
from download_nmdb import (
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_STATIONS,
    latest_complete_chunk_end,
    parse_date,
    parse_stations,
)
from station_metadata import (
    CUTOFF_RIGIDITY_GV,
    stations_by_cutoff_rigidity,
)


HISTOGRAM_BINS = 160
PLOT_GROUP_SIZE = 6


@dataclass
class DailyAccumulator:
    raw_value_sum: float = 0.0
    observed_count: int = 0
    value_sum: float = 0.0
    valid_count: int = 0
    null_count: int = 0
    highres_outlier_count: int = 0


def iter_data_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("start_date_time"):
                continue
            fields = line.split(";", 1)
            if len(fields) != 2:
                continue
            try:
                timestamp = dt.datetime.fromisoformat(fields[0].strip())
            except ValueError:
                continue
            value_text = fields[1].strip().lower()
            if value_text in NULL_VALUES:
                yield timestamp, None
                continue
            try:
                value = float(value_text)
            except ValueError:
                continue
            yield timestamp, value if math.isfinite(value) else None


def iqr_limits(values: np.ndarray, multiplier: float) -> dict[str, float]:
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower": float(q1 - multiplier * iqr),
        "upper": float(q3 + multiplier * iqr),
    }


def date_range(start: dt.date, end: dt.date):
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)


def discover_station_files(
    input_dir: Path,
    station: str,
    start: dt.date,
    end: dt.date,
) -> list[Path]:
    selected: list[tuple[dt.date, dt.date, Path]] = []
    for path in input_dir.iterdir():
        if not path.is_file() or not path.name.endswith(".txt"):
            continue
        parsed = parse_filename(path)
        if parsed is None:
            continue
        file_station, file_start, file_end, _resolution, is_no_data = parsed
        if is_no_data or file_station != station:
            continue
        if file_end < start or file_start > end:
            continue
        selected.append((file_start, file_end, path))
    selected.sort()
    return [item[2] for item in selected]


def validate_nonoverlap(paths: list[Path]) -> None:
    previous_end: dt.date | None = None
    previous_path: Path | None = None
    for path in paths:
        parsed = parse_filename(path)
        if parsed is None:
            continue
        _station, file_start, file_end, _resolution, _is_no_data = parsed
        if previous_end is not None and file_start <= previous_end:
            raise RuntimeError(
                f"overlapping chunks: {previous_path} and {path}; "
                "remove or separate duplicate downloads before preprocessing"
            )
        previous_end = file_end
        previous_path = path


def write_daily_csv(
    path: Path,
    start: dt.date,
    end: dt.date,
    daily: dict[dt.date, DailyAccumulator],
    daily_limits: dict[str, float] | None,
) -> tuple[int, int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_observed_days = 0
    retained_days = 0
    daily_outliers = 0
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "date",
                "daily_mean_raw",
                "subdaily_observed_count",
                "daily_mean_after_highres_iqr",
                "subdaily_valid_count",
                "subdaily_null_count",
                "subdaily_highres_outlier_count",
                "daily_iqr_outlier",
                "daily_value_candidate",
            ]
        )
        for day in date_range(start, end):
            accumulator = daily.get(day, DailyAccumulator())
            raw_daily_mean: float | None = None
            if accumulator.observed_count:
                raw_daily_mean = (
                    accumulator.raw_value_sum / accumulator.observed_count
                )
                raw_observed_days += 1
            if accumulator.valid_count:
                daily_mean = accumulator.value_sum / accumulator.valid_count
                retained_days += 1
                is_outlier = bool(
                    daily_limits
                    and (
                        daily_mean < daily_limits["lower"]
                        or daily_mean > daily_limits["upper"]
                    )
                )
                if is_outlier:
                    daily_outliers += 1
                writer.writerow(
                    [
                        day.isoformat(),
                        f"{raw_daily_mean:.12g}" if raw_daily_mean is not None else "",
                        accumulator.observed_count,
                        f"{daily_mean:.12g}",
                        accumulator.valid_count,
                        accumulator.null_count,
                        accumulator.highres_outlier_count,
                        int(is_outlier),
                        "" if is_outlier else f"{daily_mean:.12g}",
                    ]
                )
            else:
                writer.writerow(
                    [
                        day.isoformat(),
                        f"{raw_daily_mean:.12g}" if raw_daily_mean is not None else "",
                        accumulator.observed_count,
                        "",
                        0,
                        accumulator.null_count,
                        accumulator.highres_outlier_count,
                        0,
                        "",
                    ]
                )
    return raw_observed_days, retained_days, daily_outliers


def final_daily_value(
    accumulator: DailyAccumulator | None,
    daily_limits: dict[str, float] | None,
) -> float | None:
    if accumulator is None or not accumulator.valid_count:
        return None
    value = accumulator.value_sum / accumulator.valid_count
    if daily_limits and (
        value < daily_limits["lower"] or value > daily_limits["upper"]
    ):
        return None
    return value


def plot_final_daily_counts(
    path: Path,
    station: str,
    start: dt.date,
    end: dt.date,
    daily: dict[dt.date, DailyAccumulator],
    daily_limits: dict[str, float] | None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to create daily-count PDF plots; "
            "install it with 'python -m pip install matplotlib'"
        ) from exc

    dates: list[dt.date] = []
    values: list[float] = []
    for day in date_range(start, end):
        value = final_daily_value(daily.get(day), daily_limits)
        if value is None:
            continue
        dates.append(day)
        values.append(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10.0, 3.2))
    axis.plot(
        dates,
        values,
        linestyle="none",
        marker=".",
        markersize=2.2,
        color="#178314",
        markeredgewidth=0,
    )
    axis.set_xlim(start, end)
    axis.set_xlabel("Year")
    axis.set_ylabel("Count rate (counts/s)")
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.tick_params(direction="in", top=True, right=True)
    axis.text(
        0.02,
        0.94,
        f"{station} {CUTOFF_RIGIDITY_GV[station]:.2f} GV",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def distribution_histogram(
    values: np.ndarray,
    limits: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    threshold_width = limits["upper"] - limits["lower"]
    margin = 0.08 * threshold_width if threshold_width > 0 else 1.0
    plot_lower = min(float(np.min(values)), limits["lower"] - margin)
    robust_upper = float(np.quantile(values, 0.9999))
    plot_upper = max(robust_upper, limits["upper"] + margin)
    if plot_upper <= plot_lower:
        plot_upper = plot_lower + 1.0
    return np.histogram(
        values,
        bins=HISTOGRAM_BINS,
        range=(plot_lower, plot_upper),
    )


def plot_raw_distribution(
    path: Path,
    station: str,
    limits: dict[str, float] | None,
    histogram: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to create distribution PDF plots; "
            "install it with 'python -m pip install matplotlib'"
        ) from exc

    if histogram is None or limits is None:
        payload = {
            "histogram_counts": np.asarray([]),
            "histogram_edges": np.asarray([]),
            "lower": np.asarray(np.nan),
            "upper": np.asarray(np.nan),
        }
    else:
        payload = {
            "histogram_counts": histogram[0],
            "histogram_edges": histogram[1],
            "lower": np.asarray(limits["lower"]),
            "upper": np.asarray(limits["upper"]),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10.0, 4.8))
    plot_distribution_panel(axis, station, payload)
    figure.tight_layout()
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def write_plot_cache(
    path: Path,
    station: str,
    start: dt.date,
    end: dt.date,
    multiplier: float,
    daily: dict[dt.date, DailyAccumulator],
    daily_limits: dict[str, float] | None,
    highres_limits: dict[str, float] | None,
    histogram: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    dates: list[int] = []
    daily_values: list[float] = []
    for day in date_range(start, end):
        value = final_daily_value(daily.get(day), daily_limits)
        if value is None:
            continue
        dates.append(day.toordinal())
        daily_values.append(value)

    if histogram is None:
        histogram_counts = np.asarray([], dtype=np.int64)
        histogram_edges = np.asarray([], dtype=np.float64)
    else:
        histogram_counts, histogram_edges = histogram

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary_path,
        station=np.asarray(station),
        start=np.asarray(start.isoformat()),
        end=np.asarray(end.isoformat()),
        multiplier=np.asarray(multiplier),
        daily_dates=np.asarray(dates, dtype=np.int64),
        daily_values=np.asarray(daily_values, dtype=np.float64),
        histogram_counts=histogram_counts,
        histogram_edges=histogram_edges,
        lower=np.asarray(
            highres_limits["lower"] if highres_limits is not None else np.nan
        ),
        upper=np.asarray(
            highres_limits["upper"] if highres_limits is not None else np.nan
        ),
    )
    temporary_path.replace(path)


def load_plot_cache(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def add_station_label(axis, station: str) -> None:
    axis.text(
        0.02,
        0.94,
        f"{station} {CUTOFF_RIGIDITY_GV[station]:.2f} GV",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    axis.tick_params(direction="in", top=True, right=True)


def plot_distribution_panel(axis, station: str, payload: dict[str, object]) -> None:
    counts = np.asarray(payload["histogram_counts"])
    edges = np.asarray(payload["histogram_edges"])
    lower = float(np.asarray(payload["lower"]))
    upper = float(np.asarray(payload["upper"]))
    if counts.size and edges.size:
        axis.stairs(counts, edges, color="#26734d", linewidth=1.0)
        axis.axvline(lower, color="#b42318", linestyle="--", linewidth=1.2)
        axis.axvline(upper, color="#b42318", linestyle="--", linewidth=1.2)
        axis.set_xlim(float(edges[0]), float(edges[-1]))
        axis.set_yscale("log")
    axis.set_xlabel("Count rate (counts/s)")
    axis.set_ylabel("Entries")
    add_station_label(axis, station)


def plot_daily_panel(
    axis,
    station: str,
    payload: dict[str, object],
    mdates,
) -> None:
    ordinals = np.asarray(payload["daily_dates"], dtype=np.int64)
    values = np.asarray(payload["daily_values"], dtype=np.float64)
    dates = [dt.date.fromordinal(int(value)) for value in ordinals]
    axis.plot(
        dates,
        values,
        linestyle="none",
        marker=".",
        markersize=1.7,
        color="#178314",
        markeredgewidth=0,
    )
    start = dt.date.fromisoformat(str(np.asarray(payload["start"])))
    end = dt.date.fromisoformat(str(np.asarray(payload["end"])))
    axis.set_xlim(start, end)
    axis.set_xlabel("Year")
    axis.set_ylabel("Count rate (counts/s)")
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    add_station_label(axis, station)


def write_available_combined_plots(
    plot_dir: Path,
    stations: list[str],
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to create combined PDF plots; "
            "install it with 'python -m pip install matplotlib'"
        ) from exc

    cache_dir = plot_dir / ".combined_cache"
    outputs: list[Path] = []
    for group_start in range(0, len(stations), PLOT_GROUP_SIZE):
        group = stations[group_start : group_start + PLOT_GROUP_SIZE]
        if len(group) != PLOT_GROUP_SIZE:
            continue
        cache_paths = [cache_dir / f"{station}.npz" for station in group]
        if not all(path.is_file() for path in cache_paths):
            continue
        payloads = [load_plot_cache(path) for path in cache_paths]
        signatures = {
            (
                str(np.asarray(payload["start"])),
                str(np.asarray(payload["end"])),
                float(np.asarray(payload["multiplier"])),
            )
            for payload in payloads
        }
        if len(signatures) != 1:
            continue
        start_text, end_text, _ = signatures.pop()
        group_number = group_start // PLOT_GROUP_SIZE + 1
        date_suffix = (
            f"{start_text.replace('-', '')}_{end_text.replace('-', '')}.pdf"
        )
        distribution_path = plot_dir / (
            f"combined_iqr_group_{group_number}_{date_suffix}"
        )
        daily_path = plot_dir / (
            f"combined_daily_group_{group_number}_{date_suffix}"
        )

        figure, axes = plt.subplots(3, 2, figsize=(13.0, 10.0))
        for index, (station, payload) in enumerate(zip(group, payloads)):
            row = index // 2
            column = index % 2
            plot_distribution_panel(axes[row, column], station, payload)
        figure.tight_layout()
        temporary_path = distribution_path.with_name(
            f".{distribution_path.stem}.{os.getpid()}.tmp.pdf"
        )
        figure.savefig(temporary_path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        temporary_path.replace(distribution_path)

        figure, axes = plt.subplots(3, 2, figsize=(13.0, 10.0))
        for index, (station, payload) in enumerate(zip(group, payloads)):
            row = index // 2
            column = index % 2
            plot_daily_panel(axes[row, column], station, payload, mdates)
        figure.tight_layout()
        temporary_path = daily_path.with_name(
            f".{daily_path.stem}.{os.getpid()}.tmp.pdf"
        )
        figure.savefig(temporary_path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        temporary_path.replace(daily_path)
        outputs.extend([distribution_path, daily_path])
    return outputs


def process_station(
    station: str,
    input_dir: Path,
    data_dir: Path,
    plot_dir: Path,
    start: dt.date,
    end: dt.date,
    multiplier: float,
) -> dict:
    candidate_paths = discover_station_files(input_dir, station, start, end)
    usable_paths: list[Path] = []
    skipped_files: list[dict[str, str]] = []
    for path in candidate_paths:
        parsed = parse_filename(path)
        assert parsed is not None
        _, file_start, file_end, resolution, _ = parsed
        audit = scan_data_file(path, station, file_start, file_end, resolution)
        if audit.status == "invalid":
            skipped_files.append({"path": str(path), "reason": audit.issues})
        else:
            usable_paths.append(path)
    validate_nonoverlap(usable_paths)

    value_blocks: list[np.ndarray] = []
    total_null_values = 0
    for path in usable_paths:
        values: list[float] = []
        for timestamp, value in iter_data_rows(path):
            if timestamp.date() < start or timestamp.date() > end:
                continue
            if value is None:
                total_null_values += 1
            else:
                values.append(value)
        if values:
            value_blocks.append(np.asarray(values, dtype=np.float64))

    if not value_blocks:
        output_path = data_dir / f"{station}_daily_iqr.csv"
        plot_path = plot_dir / "daily" / f"{station}_daily_raw.pdf"
        distribution_plot_path = plot_dir / "iqr" / (
            f"{station}_native_iqr_distribution.pdf"
        )
        cache_path = plot_dir / ".combined_cache" / f"{station}.npz"
        raw_observed, retained_days, daily_outliers = write_daily_csv(
            output_path, start, end, {}, None
        )
        plot_final_daily_counts(plot_path, station, start, end, {}, None)
        plot_raw_distribution(distribution_plot_path, station, None, None)
        write_plot_cache(
            cache_path,
            station,
            start,
            end,
            multiplier,
            {},
            None,
            None,
            None,
        )
        return {
            "station": station,
            "status": "no_observed_values",
            "input_files": [str(path) for path in usable_paths],
            "skipped_files": skipped_files,
            "total_input_values": 0,
            "total_null_values": total_null_values,
            "highres_iqr": None,
            "highres_outliers": 0,
            "daily_iqr": None,
            "raw_observed_days": raw_observed,
            "observed_days_after_highres_iqr": retained_days,
            "daily_outliers": daily_outliers,
            "output": str(output_path),
            "plot": str(plot_path),
            "distribution_plot": str(distribution_plot_path),
            "plot_cache": str(cache_path),
        }

    all_values = np.concatenate(value_blocks)
    highres_limits = iqr_limits(all_values, multiplier)
    histogram = distribution_histogram(all_values, highres_limits)
    del value_blocks
    del all_values

    daily: dict[dt.date, DailyAccumulator] = {}
    highres_outliers = 0
    previous_timestamp: dt.datetime | None = None
    for path in usable_paths:
        for timestamp, value in iter_data_rows(path):
            day = timestamp.date()
            if day < start or day > end:
                continue
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise RuntimeError(
                    f"timestamps are not globally increasing at {path}: {timestamp}"
                )
            previous_timestamp = timestamp
            accumulator = daily.setdefault(day, DailyAccumulator())
            if value is None:
                accumulator.null_count += 1
            else:
                accumulator.raw_value_sum += value
                accumulator.observed_count += 1
                if value < highres_limits["lower"] or value > highres_limits["upper"]:
                    accumulator.highres_outlier_count += 1
                    highres_outliers += 1
                else:
                    accumulator.value_sum += value
                    accumulator.valid_count += 1

    daily_means = np.asarray(
        [
            accumulator.value_sum / accumulator.valid_count
            for accumulator in daily.values()
            if accumulator.valid_count
        ],
        dtype=np.float64,
    )
    daily_limits = iqr_limits(daily_means, multiplier) if daily_means.size else None
    output_path = data_dir / f"{station}_daily_iqr.csv"
    plot_path = plot_dir / "daily" / f"{station}_daily_raw.pdf"
    distribution_plot_path = (
        plot_dir / "iqr" / f"{station}_native_iqr_distribution.pdf"
    )
    cache_path = plot_dir / ".combined_cache" / f"{station}.npz"
    raw_observed_days, retained_days, daily_outliers = write_daily_csv(
        output_path, start, end, daily, daily_limits
    )
    plot_final_daily_counts(
        plot_path, station, start, end, daily, daily_limits
    )
    plot_raw_distribution(
        distribution_plot_path, station, highres_limits, histogram
    )
    write_plot_cache(
        cache_path,
        station,
        start,
        end,
        multiplier,
        daily,
        daily_limits,
        highres_limits,
        histogram,
    )
    return {
        "station": station,
        "status": "processed",
        "input_files": [str(path) for path in usable_paths],
        "skipped_files": skipped_files,
        "total_input_values": int(sum(item.observed_count for item in daily.values())),
        "total_null_values": int(sum(item.null_count for item in daily.values())),
        "highres_iqr": highres_limits,
        "highres_outliers": highres_outliers,
        "daily_iqr": daily_limits,
        "raw_observed_days": raw_observed_days,
        "observed_days_after_highres_iqr": retained_days,
        "daily_outliers": daily_outliers,
        "output": str(output_path),
        "plot": str(plot_path),
        "distribution_plot": str(distribution_plot_path),
        "plot_cache": str(cache_path),
    }


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Apply native-resolution IQR, daily means, and daily IQR."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=base / "rawdata" / "nmdb_best"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=base / "rawdata" / "nmdb_daily_iqr"
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="PDF output directory (default: OUTPUT_DIR/plots)",
    )
    parser.add_argument(
        "--stations",
        type=parse_stations,
        default=list(DEFAULT_STATIONS),
        help="comma-separated stations (default: the paper's 18 stations)",
    )
    parser.add_argument("--start", type=parse_date, default=DEFAULT_START)
    end_selection = parser.add_mutually_exclusive_group()
    end_selection.add_argument("--end", type=parse_date)
    end_selection.add_argument(
        "--latest",
        action="store_true",
        help="process through the latest complete --chunk-months block",
    )
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=3,
        help="months per download block when using --latest (default: 3)",
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=3.0,
        help="IQR multiplier used at native and daily resolution (default: 3)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.chunk_months < 1:
        raise SystemExit("--chunk-months must be at least 1")
    if args.latest:
        try:
            args.end = latest_complete_chunk_end(
                args.start, args.chunk_months
            )
        except ValueError as exc:
            raise SystemExit(f"--latest: {exc}") from exc
    elif args.end is None:
        args.end = DEFAULT_END
    if args.start > args.end:
        raise SystemExit("--start must not be after --end")
    if args.iqr_multiplier <= 0:
        raise SystemExit("--iqr-multiplier must be positive")
    if not args.input_dir.is_dir():
        raise SystemExit(f"input directory does not exist: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.plot_dir or args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    (plot_dir / "daily").mkdir(parents=True, exist_ok=True)
    (plot_dir / "iqr").mkdir(parents=True, exist_ok=True)

    missing_metadata = sorted(set(args.stations) - CUTOFF_RIGIDITY_GV.keys())
    if missing_metadata:
        raise SystemExit(
            "missing cutoff-rigidity metadata for: " + ", ".join(missing_metadata)
        )

    reports: list[dict] = []
    for index, station in enumerate(args.stations, start=1):
        print(f"[{index}/{len(args.stations)}] process {station}")
        report = process_station(
            station,
            args.input_dir,
            data_dir,
            plot_dir,
            args.start,
            args.end,
            args.iqr_multiplier,
        )
        reports.append(report)
        print(
            f"  {report['status']}: raw_observed_days={report['raw_observed_days']}, "
            f"retained_days={report['observed_days_after_highres_iqr']}, "
            f"highres_outliers={report['highres_outliers']}, "
            f"daily_outliers={report['daily_outliers']}"
        )
        print(f"  plot: {report['plot']}")
        print(f"  distribution: {report['distribution_plot']}")

    combined_plots = write_available_combined_plots(
        plot_dir, stations_by_cutoff_rigidity(list(DEFAULT_STATIONS))
    )
    for path in combined_plots:
        print(f"Combined plot: {path}")

    summary = {
        "date_range": [args.start.isoformat(), args.end.isoformat()],
        "iqr_multiplier": args.iqr_multiplier,
        "stations": reports,
        "combined_plots": [str(path) for path in combined_plots],
        "notes": [
            "daily_iqr_outlier is a statistical candidate, not yet a confirmed bad value",
            "cross-station solar-event restoration must run before final removal",
            "no minimum intraday coverage threshold is applied because the paper does not specify one",
        ],
    }
    summary_path = args.output_dir / "preprocess_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
