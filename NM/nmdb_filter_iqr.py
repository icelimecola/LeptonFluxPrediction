#!/usr/bin/env python3
"""Apply two-stage 3*IQR filtering and compute daily NM count rates."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from audit_nmdb import (
    NULL_VALUES,
    FileAudit,
    merge_duplicate_timestamps,
    parse_filename,
    scan_data_file,
)
from nmdb_download import (
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
# Values are flushed to numpy in blocks of this size so that merging the
# preferred and fallback sources does not hold the whole series as Python floats.
VALUE_BLOCK_FLUSH = 1_000_000
# NMDB table whose samples are drawn in green in the raw-resolution plot, so the
# days filled from the hourly companion stay separable from the minute-resolution
# revori ones. Kept here (not in the test script) so production and the A/B test
# cannot drift apart on colours.
HOURLY_TABLE = "1 hour validated"
HOURLY_SAMPLE_COLOR = "C2"            # raw 1h samples in the background cloud
HOURLY_REJECTED_COLOR = "darkgreen"   # 1h samples dropped by the IQR band
REVORI_SAMPLE_COLOR = "0.75"          # raw revori samples (unchanged grey)
REVORI_REJECTED_COLOR = "C3"          # revori samples dropped by the band
DISPLAY_POINTS = 150_000              # cap for the subsampled background cloud
MAX_REJECTED_STORED = 400_000         # safety cap on the stored crosses
# Fixed seed so the subsampled scatter in the raw-resolution plot is reproducible
# across runs (the plot is a diagnostic, not a statistic).
SAMPLE_SEED = 17


@dataclass
class DailyAccumulator:
    raw_value_sum: float = 0.0
    observed_count: int = 0
    value_sum: float = 0.0
    valid_count: int = 0
    null_count: int = 0
    highres_outlier_count: int = 0
    # NMDB tables that fed this day. The merge rule guarantees one source per
    # day, so this normally holds exactly one name; more than one means the
    # merge was bypassed somewhere and is worth surfacing in the CSV.
    sources: set[str] = field(default_factory=set)


def _iter_raw_rows(path: Path):
    """Yield ``(timestamp, value | None)`` for every data row of one chunk."""
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


def iter_data_rows(path: Path, *, merge_duplicates: bool = True):
    """Read one chunk, merging NMDB's repeated timestamps by default.

    The merge is on by default rather than behind a flag: it is a repair of a
    known upstream artefact (see ``audit_nmdb.merge_duplicate_timestamps``), it
    is an identity for the identical-value repeats, and leaving it optional
    would mean that forgetting the flag silently drops every affected quarter
    without any error.
    """
    rows = _iter_raw_rows(path)
    if merge_duplicates:
        return merge_duplicate_timestamps(rows)
    return rows


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
    # Recursive: downloads live one subdirectory per station, while a flat
    # directory from an older run still works.
    for path in input_dir.rglob("*.txt"):
        if not path.is_file():
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


def discover_station_sources(
    input_dirs: Sequence[Path],
    station: str,
    start: dt.date,
    end: dt.date,
) -> list[list[Path]]:
    """One sorted file list per input directory, in priority order.

    Index 0 is the preferred table (``revori``); the directories after it only
    supply the days index 0 leaves empty. See ``iter_merged_rows``.
    """
    return [
        discover_station_files(directory, station, start, end)
        for directory in input_dirs
    ]


@dataclass(frozen=True)
class SourceFile:
    """One audited chunk: its window, resolution group, table and raw header."""

    start: dt.date
    end: dt.date
    group: str
    table: str
    raw_header: str


def audit_station_sources(
    input_dirs: Sequence[Path],
    station: str,
    start: dt.date,
    end: dt.date,
) -> tuple[
    list[list[Path]],
    list[list[SourceFile]],
    list[list[FileAudit]],
    list[dict[str, str]],
]:
    """Audit every chunk of one station across the priority-ordered directories.

    Returns ``(usable, specs, audits, skipped)``, all parallel: level 0 is the
    preferred table and each later level only supplies the whole days the
    earlier ones are missing (see ``iter_merged_rows``). A level with no usable
    file is dropped, so an index means the same thing in all four lists.

    The grouping key is the audit's normalised ``effective_resolution_minutes``,
    NOT the raw ``ORIGINAL RES`` header string: the header is unreliable (AATB
    writes ``multiple: min = 0 min, max = 1 min`` on files whose sampling is in
    fact a uniform 60 s) and would create spurious groups. The raw header is
    kept in ``SourceFile.raw_header`` for reporting only.
    """
    usable_levels: list[list[Path]] = []
    spec_levels: list[list[SourceFile]] = []
    audit_levels: list[list[FileAudit]] = []
    skipped: list[dict[str, str]] = []
    for paths in discover_station_sources(input_dirs, station, start, end):
        usable: list[Path] = []
        level_specs: list[SourceFile] = []
        level_audits: list[FileAudit] = []
        for path in paths:
            parsed = parse_filename(path)
            assert parsed is not None
            _, file_start, file_end, resolution, _ = parsed
            audit = scan_data_file(path, station, file_start, file_end, resolution)
            if audit.status == "invalid":
                skipped.append({"path": str(path), "reason": audit.issues})
                continue
            usable.append(path)
            level_audits.append(audit)
            level_specs.append(
                SourceFile(
                    start=file_start,
                    end=file_end,
                    group=f"{audit.effective_resolution_minutes} min",
                    table=audit.nmdb_table.strip(),
                    raw_header=audit.original_resolution.strip() or "unknown",
                )
            )
        validate_nonoverlap(usable)
        if usable:
            usable_levels.append(usable)
            spec_levels.append(level_specs)
            audit_levels.append(level_audits)
    return usable_levels, spec_levels, audit_levels, skipped


def spec_of(
    specs: Sequence[Sequence[SourceFile]], level: int, day: dt.date
) -> SourceFile | None:
    """The audited chunk that served ``day`` at ``level``.

    Chunks inside one level never overlap, so a binary search over the sorted
    windows is enough.
    """
    windows = specs[level]
    lo, hi = 0, len(windows) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        spec = windows[mid]
        if day < spec.start:
            hi = mid - 1
        elif day > spec.end:
            lo = mid + 1
        else:
            return spec
    return None


def group_of(
    specs: Sequence[Sequence[SourceFile]], level: int, day: dt.date
) -> str | None:
    """Resolution group that served ``day`` at ``level``."""
    spec = spec_of(specs, level, day)
    return spec.group if spec is not None else None


def group_spans(
    specs: Sequence[Sequence[SourceFile]],
) -> dict[str, tuple[dt.date, dt.date]]:
    """First and last day each resolution group covers."""
    spans: dict[str, list[dt.date]] = collections.defaultdict(list)
    for level_specs in specs:
        for spec in level_specs:
            spans[spec.group].extend([spec.start, spec.end])
    return {group: (min(v), max(v)) for group, v in spans.items()}


def group_meta(
    specs: Sequence[Sequence[SourceFile]],
) -> dict[str, dict[str, list[str]]]:
    """Raw headers and NMDB tables that fed each resolution group."""
    meta: dict[str, dict[str, list[str]]] = {}
    for level_specs in specs:
        for spec in level_specs:
            entry = meta.setdefault(
                spec.group, {"raw_headers": [], "nmdb_tables": []}
            )
            if spec.raw_header not in entry["raw_headers"]:
                entry["raw_headers"].append(spec.raw_header)
            if spec.table not in entry["nmdb_tables"]:
                entry["nmdb_tables"].append(spec.table)
    return {
        group: {key: sorted(values) for key, values in entry.items()}
        for group, entry in meta.items()
    }


def iter_merged_rows(
    sources: Sequence[Sequence[Path]],
    start: dt.date,
    end: dt.date,
    *,
    merge_duplicates: bool = True,
) -> Iterator[tuple[dt.datetime, float | None, int]]:
    """Yield ``(timestamp, value | None, source_level)`` merged across sources.

    The merging rule, in the words of the paper's data policy:

    * a day is served by exactly one source -- the first one that has any real
      value for it -- so **one day never mixes two resolutions**;
    * **every** value of the preferred source is kept;
    * the fallback source contributes only whole days the preferred source is
      missing entirely.

    A day is buffered until it has been read completely, because whether it is
    "missing" is only known then: a day whose rows are all nulls carries no
    observation, so it is left for the next source instead of blocking it.
    ``source_level`` is 0 for the preferred source and >0 for a filled day, so
    callers can report how much of the series is fallback data.

    Timestamps are checked for monotonicity per file rather than globally: after
    merging, the sources are no longer globally ordered (the fallback supplies
    days *inside* the preferred source's range), and nothing downstream depends
    on that order -- quantiles and daily sums are both order-independent.
    """
    covered: set[dt.date] = set()
    for level, paths in enumerate(sources):
        buffer: list[tuple[dt.datetime, float | None]] = []
        buffer_day: dt.date | None = None
        has_value = False
        for path in paths:
            previous: dt.datetime | None = None
            for timestamp, value in iter_data_rows(
                path, merge_duplicates=merge_duplicates
            ):
                if previous is not None and timestamp <= previous:
                    raise RuntimeError(
                        f"timestamps are not increasing at {path}: {timestamp}"
                    )
                previous = timestamp
                day = timestamp.date()
                if day < start or day > end:
                    continue
                if level and day in covered:
                    continue
                if day != buffer_day:
                    if has_value:
                        covered.add(buffer_day)
                        for row in buffer:
                            yield row[0], row[1], level
                    buffer = []
                    has_value = False
                    buffer_day = day
                buffer.append((timestamp, value))
                if value is not None:
                    has_value = True
        if has_value:
            covered.add(buffer_day)
            for row in buffer:
                yield row[0], row[1], level


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
                "subdaily_source",
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
                        "+".join(sorted(accumulator.sources)),
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
                        "+".join(sorted(accumulator.sources)),
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
    sample_points: Sequence[tuple[dt.datetime, float, str]] = (),
    rejected_points: Sequence[tuple[dt.datetime, float, str]] = (),
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
        # The raw-series points are cached too, because the combined figure is
        # drawn from these files in a separate pass and re-reading 5 GB of chunks
        # just to draw a scatter plot would be absurd.
        raw_sample_seconds=_epoch_seconds([p[0] for p in sample_points]),
        raw_sample_values=np.asarray([p[1] for p in sample_points], dtype=np.float64),
        raw_sample_hourly=_hourly_flags(sample_points),
        raw_rejected_seconds=_epoch_seconds([p[0] for p in rejected_points]),
        raw_rejected_values=np.asarray(
            [p[1] for p in rejected_points], dtype=np.float64
        ),
        raw_rejected_hourly=_hourly_flags(rejected_points),
    )
    temporary_path.replace(path)


def _hourly_flags(
    points: Sequence[tuple[dt.datetime, float, str]],
) -> np.ndarray:
    return np.asarray(
        [1 if p[2].strip().lower() == HOURLY_TABLE else 0 for p in points],
        dtype=np.uint8,
    )


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


_EPOCH = dt.datetime(1970, 1, 1)


def _epoch_seconds(timestamps) -> np.ndarray:
    """Naive UTC datetimes -> seconds since the epoch (no local-time shifting)."""
    return np.asarray(
        [(stamp - _EPOCH).total_seconds() for stamp in timestamps], dtype=np.float64
    )


def _epoch_datetimes(seconds) -> list[dt.datetime]:
    return [_EPOCH + dt.timedelta(seconds=float(value)) for value in seconds]


def split_by_table(
    points: Sequence[tuple[dt.datetime, float, str]],
) -> tuple[list[tuple[dt.datetime, float]], list[tuple[dt.datetime, float]]]:
    """Split ``(timestamp, value, table)`` points into (revori, hourly) lists."""
    revori: list[tuple[dt.datetime, float]] = []
    hourly: list[tuple[dt.datetime, float]] = []
    for timestamp, value, table in points:
        if table.strip().lower() == HOURLY_TABLE:
            hourly.append((timestamp, value))
        else:
            revori.append((timestamp, value))
    return revori, hourly


def plot_subdaily_series(
    path: Path,
    station: str,
    limits: dict[str, float],
    sample_points: Sequence[tuple[dt.datetime, float, str]],
    rejected_points: Sequence[tuple[dt.datetime, float, str]],
    *,
    n_fill_days: int,
    n_rejected: int,
    n_rejected_hourly: int,
) -> None:
    """Raw sub-daily samples vs time, with the IQR-rejected ones crossed.

    Same visual language as the A/B test script's panel A, so the two can be read
    side by side: grey = revori (subsampled), green = 1h samples on the days
    revori is missing (kept in full), red crosses = rejected revori, dark-green
    crosses = rejected 1h.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    revori_sample, hourly_sample = split_by_table(sample_points)
    revori_rejected, hourly_rejected = split_by_table(rejected_points)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(13.0, 5.4))
    if revori_sample:
        axis.plot([p[0] for p in revori_sample], [p[1] for p in revori_sample],
                  linestyle="none", marker=".", markersize=0.7,
                  color=REVORI_SAMPLE_COLOR, markeredgewidth=0,
                  label=f"revori samples, subsampled (n={len(revori_sample):,})")
    if hourly_sample:
        axis.plot([p[0] for p in hourly_sample], [p[1] for p in hourly_sample],
                  linestyle="none", marker=".", markersize=1.3,
                  color=HOURLY_SAMPLE_COLOR, markeredgewidth=0,
                  label=f"1h samples, filled days, all (n={len(hourly_sample):,})")
    if revori_rejected:
        axis.plot([p[0] for p in revori_rejected],
                  [p[1] for p in revori_rejected],
                  linestyle="none", marker="x", markersize=4.2,
                  markeredgewidth=0.9, color=REVORI_REJECTED_COLOR,
                  label=f"IQR-rejected revori (n={len(revori_rejected):,})")
    if hourly_rejected:
        axis.plot([p[0] for p in hourly_rejected],
                  [p[1] for p in hourly_rejected],
                  linestyle="none", marker="x", markersize=4.6,
                  markeredgewidth=1.0, color=HOURLY_REJECTED_COLOR,
                  label=f"IQR-rejected 1h (n={len(hourly_rejected):,})")
    axis.axhline(limits["lower"], color="C0", ls="--", lw=1.0,
                 label=f"band [{limits['lower']:.2f}, {limits['upper']:.2f}]")
    axis.axhline(limits["upper"], color="C0", ls="--", lw=1.0)
    axis.set_title(f"{station} — raw sub-daily samples, pooled band "
                   f"({station} Rc={CUTOFF_RIGIDITY_GV[station]:g} GV)",
                   fontsize=11, fontweight="bold")
    axis.set_ylabel("count rate (counts/s)")
    axis.set_xlabel("time (UTC)")
    axis.tick_params(direction="in", top=True, right=True)
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.legend(
        loc="best",
        fontsize=8,
        markerscale=3,
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
    )
    figure.suptitle(
        f"{station} — sub-daily values and the pooled 3*IQR band"
        f"      days filled from 1h: {n_fill_days}      "
        f"rejected: {n_rejected:,} (of which 1h: {n_rejected_hourly:,})\n"
        f"grey dots = revori (subsampled), green dots = 1h fill (complete);   "
        f"red crosses = rejected revori, dark-green crosses = rejected 1h",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def plot_raw_panel(axis, station: str, payload: dict[str, object], mdates) -> None:
    """One panel of the combined raw-series figure, drawn from the plot cache."""
    seconds = np.asarray(payload.get("raw_sample_seconds", []), dtype=np.float64)
    values = np.asarray(payload.get("raw_sample_values", []), dtype=np.float64)
    hourly = np.asarray(payload.get("raw_sample_hourly", []), dtype=np.uint8)
    if seconds.size and values.size == seconds.size:
        dates = _epoch_datetimes(seconds)
        revori_index = [i for i, flag in enumerate(hourly) if not flag]
        hourly_index = [i for i, flag in enumerate(hourly) if flag]
        if revori_index:
            axis.plot([dates[i] for i in revori_index],
                      [values[i] for i in revori_index],
                      linestyle="none", marker=".", markersize=0.5,
                      color=REVORI_SAMPLE_COLOR, markeredgewidth=0)
        if hourly_index:
            axis.plot([dates[i] for i in hourly_index],
                      [values[i] for i in hourly_index],
                      linestyle="none", marker=".", markersize=1.0,
                      color=HOURLY_SAMPLE_COLOR, markeredgewidth=0)
    lower = float(np.asarray(payload.get("lower", np.nan)))
    upper = float(np.asarray(payload.get("upper", np.nan)))
    if np.isfinite(lower):
        axis.axhline(lower, color="C0", ls="--", lw=0.8)
    if np.isfinite(upper):
        axis.axhline(upper, color="C0", ls="--", lw=0.8)
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
        raw_path = plot_dir / (
            f"combined_raw_group_{group_number}_{date_suffix}"
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

        figure, axes = plt.subplots(3, 2, figsize=(13.0, 10.0))
        for index, (station, payload) in enumerate(zip(group, payloads)):
            row = index // 2
            column = index % 2
            plot_raw_panel(axes[row, column], station, payload, mdates)
        figure.tight_layout()
        temporary_path = raw_path.with_name(
            f".{raw_path.stem}.{os.getpid()}.tmp.pdf"
        )
        figure.savefig(temporary_path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        temporary_path.replace(raw_path)
        outputs.extend([distribution_path, daily_path, raw_path])
    return outputs


def process_station(
    station: str,
    input_dirs: Sequence[Path],
    data_dir: Path,
    plot_dir: Path,
    start: dt.date,
    end: dt.date,
    multiplier: float,
) -> dict:
    """Filter one station over all priority-ordered input directories.

    ``input_dirs[0]`` is the preferred table and the rest fill its missing days
    at whole-day granularity (see ``iter_merged_rows``). Overlap is only
    forbidden *within* one directory: the same quarter appearing in the
    preferred and the fallback directory is the intended layout.
    """
    usable_sources, specs, level_audits, skipped_files = audit_station_sources(
        input_dirs, station, start, end
    )
    tables_used: Counter = Counter()
    duplicates_merged = 0
    for audits in level_audits:
        for audit in audits:
            tables_used[audit.nmdb_table or "<missing>"] += 1
            duplicates_merged += audit.duplicate_timestamps
    input_files = [path for paths in usable_sources for path in paths]

    value_blocks: list[np.ndarray] = []
    total_null_values = 0
    fill_days: set[dt.date] = set()
    buffer: list[float] = []
    for timestamp, value, level in iter_merged_rows(usable_sources, start, end):
        if level:
            fill_days.add(timestamp.date())
        if value is None:
            total_null_values += 1
            continue
        buffer.append(value)
        if len(buffer) >= VALUE_BLOCK_FLUSH:
            value_blocks.append(np.asarray(buffer, dtype=np.float64))
            buffer = []
    if buffer:
        value_blocks.append(np.asarray(buffer, dtype=np.float64))

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
            "input_files": [str(path) for path in input_files],
            "input_dirs": [str(directory) for directory in input_dirs],
            "preferred_files": int(len(usable_sources[0])) if usable_sources else 0,
            "fallback_files": int(sum(len(s) for s in usable_sources[1:])),
            "days_filled_from_fallback": int(len(fill_days)),
            "skipped_files": skipped_files,
            "tables_used": dict(tables_used),
            "duplicates_merged": duplicates_merged,
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
    total_values = int(all_values.size)
    highres_limits = iqr_limits(all_values, multiplier)
    histogram = distribution_histogram(all_values, highres_limits)
    del value_blocks
    del all_values

    daily: dict[dt.date, DailyAccumulator] = {}
    highres_outliers = 0
    hourly_outliers = 0
    sample_points: list[tuple[dt.datetime, float, str]] = []
    rejected_points: list[tuple[dt.datetime, float, str]] = []
    sample_probability = min(1.0, DISPLAY_POINTS / max(total_values, 1))
    sample_rng = np.random.default_rng(SAMPLE_SEED)
    for timestamp, value, level in iter_merged_rows(usable_sources, start, end):
        day = timestamp.date()
        spec = spec_of(specs, level, day)
        table = spec.table if spec is not None else "<unknown>"
        hourly = table.strip().lower() == HOURLY_TABLE
        accumulator = daily.setdefault(day, DailyAccumulator())
        accumulator.sources.add(table)
        if value is None:
            accumulator.null_count += 1
            continue
        accumulator.raw_value_sum += value
        accumulator.observed_count += 1
        if value < highres_limits["lower"] or value > highres_limits["upper"]:
            accumulator.highres_outlier_count += 1
            highres_outliers += 1
            if hourly:
                hourly_outliers += 1
            if len(rejected_points) < MAX_REJECTED_STORED:
                rejected_points.append((timestamp, value, table))
        else:
            accumulator.value_sum += value
            accumulator.valid_count += 1
        # The hourly cloud is kept in full and only the minute-resolution one is
        # subsampled: the filled days are a fraction of a percent of all values,
        # so a shared subsampling rate would leave them invisible in the plot.
        if hourly or sample_probability >= 1.0 or sample_rng.random() < sample_probability:
            sample_points.append((timestamp, value, table))

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
    raw_plot_path = plot_dir / "raw" / f"{station}_subdaily_raw.pdf"
    raw_observed_days, retained_days, daily_outliers = write_daily_csv(
        output_path, start, end, daily, daily_limits
    )
    plot_final_daily_counts(
        plot_path, station, start, end, daily, daily_limits
    )
    plot_raw_distribution(
        distribution_plot_path, station, highres_limits, histogram
    )
    plot_subdaily_series(
        raw_plot_path,
        station,
        highres_limits,
        sample_points,
        rejected_points,
        n_fill_days=len(fill_days),
        n_rejected=highres_outliers,
        n_rejected_hourly=hourly_outliers,
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
        sample_points=sample_points,
        rejected_points=rejected_points,
    )
    return {
        "station": station,
        "status": "processed",
        "input_files": [str(path) for path in input_files],
        "input_dirs": [str(directory) for directory in input_dirs],
        "preferred_files": int(len(usable_sources[0])) if usable_sources else 0,
        "fallback_files": int(sum(len(s) for s in usable_sources[1:])),
        "days_filled_from_fallback": int(len(fill_days)),
        "skipped_files": skipped_files,
        "tables_used": dict(tables_used),
        "duplicates_merged": duplicates_merged,
        "total_input_values": int(sum(item.observed_count for item in daily.values())),
        "total_null_values": int(sum(item.null_count for item in daily.values())),
        "highres_iqr": highres_limits,
        "highres_outliers": highres_outliers,
        "highres_outliers_hourly": hourly_outliers,
        "daily_iqr": daily_limits,
        "raw_observed_days": raw_observed_days,
        "observed_days_after_highres_iqr": retained_days,
        "daily_outliers": daily_outliers,
        "output": str(output_path),
        "plot": str(plot_path),
        "distribution_plot": str(distribution_plot_path),
        "raw_plot": str(raw_plot_path),
        "plot_cache": str(cache_path),
    }


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Apply native-resolution IQR, daily means, and daily IQR."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        help=(
            "directory holding downloaded chunks; repeat the flag to declare a "
            "fallback chain, e.g. --input-dir data/nmdb_best_revori "
            "--input-dir data/nmdb_best_1h. The first directory is the "
            "preferred table and every later one only fills the whole days it "
            "is missing, so a day is never assembled from two resolutions "
            "(default: data/nmdb_best_revori then data/nmdb_best_1h)"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=base / "data" / "nmdb_daily_iqr"
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
    if not args.input_dir:
        args.input_dir = [
            Path(__file__).resolve().parent / "data" / "nmdb_best_revori",
            Path(__file__).resolve().parent / "data" / "nmdb_best_1h",
        ]
    for directory in args.input_dir:
        if not directory.is_dir():
            raise SystemExit(f"input directory does not exist: {directory}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.plot_dir or args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    (plot_dir / "daily").mkdir(parents=True, exist_ok=True)
    (plot_dir / "iqr").mkdir(parents=True, exist_ok=True)
    (plot_dir / "raw").mkdir(parents=True, exist_ok=True)

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
            "KNOWN, NOT YET FIXED: the sub-daily 3*IQR band also rejects the deepest "
            "part of genuine Forbush decreases, which inflates that day's mean. "
            "Measured on 2003-10-29 (the Halloween event): AATB raw 1190.04 -> "
            "1231.73 (+41.69), MXCO 186.64 -> 201.10 (+14.45), JUNG 127.57 -> "
            "138.11 (+10.54). This is left to the cross-station solar-event "
            "restoration step above; a single-station band cannot tell a sustained "
            "physical depression from an isolated spike",
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
