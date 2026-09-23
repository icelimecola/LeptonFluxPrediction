#!/usr/bin/env python3
"""TEST SCRIPT -- compare two IQR strategies on mixed-resolution stations.

This file is deliberately standalone: it does NOT modify ``nmdb_filter_iqr.py``
and it only ever writes to ``*_test`` paths.

Input
    The same priority chain as the production filter: ``--input-dir`` may be
    repeated, the first directory is the preferred table (``revori``) and every
    later one only fills the whole days it is missing. The merged stream is built
    by ``nmdb_filter_iqr.iter_merged_rows`` itself, so method A below reproduces
    the production band *exactly* -- that is what makes the comparison meaningful.

Method A -- the current pipeline
    Pool every sub-daily value of the station over the whole ``[start, end]``
    range into one array, compute a single ``3*IQR`` band, and drop every value
    outside it.

Method B -- resolution-aware
    Group the sub-daily values by the native resolution of the file that
    supplied the day -- 1 min for a ``revori`` day, 60 min for a day filled from
    the ``1h`` companion -- compute one ``3*IQR`` band *per group*, and drop a
    value using its own group's band. Surviving values are then merged into daily
    means exactly as in method A.

    The daily-level (second) IQR stage is still pooled in both methods: every
    daily mean belongs to one resolution group anyway, and keeping that stage
    identical isolates the effect being measured.

Motivation
    Two different things put more than one sampling resolution inside one station
    series:

    * the archive history -- JUNG / JUNG1 are 60 min for 2001-2008-08 and 1 min
      after; INVK and THUL are 60 min for part of 2001; MXCO is 5 min before
      2008. For MXCO both eras come from the SAME NMDB table ("revised
      original"), i.e. the table choice does not determine the resolution;
    * the day-level fill -- every day that ``revori`` is missing entirely is
      supplied by the hourly ``1h`` companion, so those days carry 24 values
      instead of 1440.

    A 5-minute, a 1-minute and a 60-minute sample do not have the same spread, so
    one pooled band is a compromise that fits none of them equally. This script
    measures how much that actually matters, and plots the raw series with the
    rejected samples marked as red crosses.

Reading the result -- one caveat
    The filled days are NOT spread evenly over time: they cluster where the
    archive is thin (JUNG / JUNG1 2001-2008, MXCO 2010, TERA 2001-2005, AATB in
    scattered quarters). Grouping by resolution therefore also groups by period,
    so a per-group band can absorb a solar-cycle-level difference in count rate
    and not only a sampling difference. When a group's band looks much narrower
    or wider than the pooled one, check the group's span before attributing the
    whole gap to resolution -- compare it with the same period's values inside
    the pooled group first.

Outputs (all suffixed ``_test``, under ``data/nmdb_daily_iqr_test``)
    data/<ST>_daily_iqr_A_test.csv     method A daily series (production columns)
    data/<ST>_daily_iqr_B_test.csv     method B daily series
    plots/<ST>_iqr_resolution_compare_test.pdf
                                       raw sub-daily samples vs time with the
                                       resolution groups, each method's band, and
                                       the rejected samples as red crosses
    plots/<ST>_daily_AB_test.pdf       daily series A vs B plus their difference
    summary_test.json                  bands, rejection counts, day-level diffs
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audit_nmdb import parse_filename, scan_data_file
from nmdb_download import parse_date, parse_stations
from nmdb_filter_iqr import (
    DailyAccumulator,
    VALUE_BLOCK_FLUSH,
    discover_station_sources,
    iqr_limits,
    iter_merged_rows,
    validate_nonoverlap,
    write_daily_csv,
)

BASE = Path(__file__).resolve().parent
DEFAULT_START = dt.date(2001, 1, 1)
DEFAULT_END = dt.date(2026, 6, 30)
# Stations that genuinely change sampling resolution inside the recording:
#   JUNG / JUNG1 : 60 min (1h table) 2001-2008-08, then 1 min (revori)
#   INVK         : 60 min for 2001-01..2001-09
#   THUL         : 60 min for 2001-01..2001-06
#   MXCO         : 5 min 2001-01..2008-08, then 1 min -- both from the *same*
#                  NMDB table, i.e. the table does not determine the resolution
#   AATB         : 1 min throughout, but 280 days are filled from the hourly
#                  companion, so it now has a small 60 min group too; its header
#                  also reads "multiple: min = 0 min, max = 1 min" on two files
#                  whose sampling is in fact a uniform 60 s, which must NOT
#                  create a spurious group.
# NOTE the pair to sanity-check the script itself is a station with no filled
# days (OULU, FSMT): there the band is pooled over one resolution only, so A and
# B must come out bit-identical. AATB can no longer serve as that control.
DEFAULT_TEST_STATIONS = ["JUNG", "JUNG1", "INVK", "THUL", "MXCO", "AATB"]
DISPLAY_POINTS = 150_000      # cap for the grey background cloud
MAX_REJECTED_STORED = 400_000  # safety cap on the red crosses
# NMDB table whose samples are drawn in green, so the days filled from the
# hourly companion are visually separable from the minute-resolution revori ones.
HOURLY_TABLE = "1 hour validated"
HOURLY_SAMPLE_COLOR = "C2"           # raw 1h samples in the background cloud
HOURLY_REJECTED_COLOR = "darkgreen"  # 1h samples dropped by the IQR band


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SourceFile:
    """One audited chunk: window, resolution group, table and raw header."""

    start: dt.date
    end: dt.date
    group: str
    table: str
    raw_header: str


def load_sources(
    input_dirs: Sequence[Path],
    station: str,
    start: dt.date,
    end: dt.date,
) -> tuple[list[list[Path]], list[list[SourceFile]], list[dict[str, str]]]:
    """Audit the priority-ordered input directories.

    Mirrors ``nmdb_filter_iqr.process_station``: ``input_dirs[0]`` is the
    preferred table and every later directory supplies only the whole days the
    earlier ones are missing. Returns ``(usable_sources, specs, skipped)``,
    where ``specs`` has one entry per level, parallel to ``usable_sources``.

    The grouping key is the audit's normalised ``effective_resolution_minutes``,
    NOT the raw ``ORIGINAL RES`` header string. The header is unreliable for
    grouping: AATB writes ``multiple: min = 0 min, max = 1 min`` on some files
    whose sampling is in fact a uniform 60 s, which would otherwise create
    spurious resolution groups. The raw header is kept for reporting only.
    """
    sources = discover_station_sources(input_dirs, station, start, end)
    usable_sources: list[list[Path]] = []
    specs: list[list[SourceFile]] = []
    skipped: list[dict[str, str]] = []
    for paths in sources:
        usable: list[Path] = []
        level_specs: list[SourceFile] = []
        for path in paths:
            parsed = parse_filename(path)
            assert parsed is not None
            _, file_start, file_end, resolution, _ = parsed
            audit = scan_data_file(path, station, file_start, file_end, resolution)
            if audit.status == "invalid":
                skipped.append({"path": str(path), "reason": audit.issues})
                continue
            usable.append(path)
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
            usable_sources.append(usable)
            specs.append(level_specs)
    return usable_sources, specs, skipped


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


def group_values(
    usable_sources: Sequence[Sequence[Path]],
    specs: Sequence[Sequence[SourceFile]],
    start: dt.date,
    end: dt.date,
) -> dict[str, np.ndarray]:
    """Pool the valid values of each resolution group over the merged stream."""
    blocks: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    buffer: dict[str, list[float]] = collections.defaultdict(list)
    for timestamp, value, level in iter_merged_rows(usable_sources, start, end):
        if value is None:
            continue
        group = group_of(specs, level, timestamp.date())
        if group is None:
            continue
        pending = buffer[group]
        pending.append(value)
        if len(pending) >= VALUE_BLOCK_FLUSH:
            blocks[group].append(np.asarray(pending, dtype=np.float64))
            buffer[group] = []
    for group, pending in buffer.items():
        if pending:
            blocks[group].append(np.asarray(pending, dtype=np.float64))
    return {group: np.concatenate(parts) for group, parts in blocks.items()}


def group_spans(
    specs: Sequence[Sequence[SourceFile]],
) -> dict[str, tuple[dt.date, dt.date]]:
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


def accumulate_both(
    usable_sources: Sequence[Sequence[Path]],
    specs: Sequence[Sequence[SourceFile]],
    start: dt.date,
    end: dt.date,
    band_a: dict[str, float],
    bands_b: dict[str, dict[str, float]],
    sample_probability: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    """One pass over the merged stream applying both methods simultaneously.

    Also collects the ``(timestamp, value, group)`` of every rejected sample for
    the plots, a Bernoulli subsample of all samples for the grey cloud, and the
    set of days that came from a fallback level.
    """
    daily_a: dict[dt.date, DailyAccumulator] = {}
    daily_b: dict[dt.date, DailyAccumulator] = {}
    rejected_a: list[tuple[dt.datetime, float, str, str]] = []
    rejected_b: list[tuple[dt.datetime, float, str, str]] = []
    # (timestamp, value, resolution group, NMDB table) -- the table is what the
    # plot colours by, so the "1 hour validated" cloud can be told apart from
    # the "revised original" one.
    sample: list[tuple[dt.datetime, float, str, str]] = []
    fill_days: set[dt.date] = set()
    n_a = n_b = 0
    n_a_hourly = n_b_hourly = 0
    lo_a, hi_a = band_a["lower"], band_a["upper"]

    for timestamp, value, level in iter_merged_rows(usable_sources, start, end):
        day = timestamp.date()
        spec = spec_of(specs, level, day)
        if spec is None:
            continue
        group = spec.group
        hourly = spec.table.strip().lower() == HOURLY_TABLE
        if level:
            fill_days.add(day)
        limits_b = bands_b[group]
        lo_b, hi_b = limits_b["lower"], limits_b["upper"]
        acc_a = daily_a.setdefault(day, DailyAccumulator())
        acc_b = daily_b.setdefault(day, DailyAccumulator())
        if value is None:
            acc_a.null_count += 1
            acc_b.null_count += 1
            continue
        acc_a.raw_value_sum += value
        acc_a.observed_count += 1
        acc_b.raw_value_sum += value
        acc_b.observed_count += 1

        if value < lo_a or value > hi_a:
            acc_a.highres_outlier_count += 1
            n_a += 1
            if hourly:
                n_a_hourly += 1
            if len(rejected_a) < MAX_REJECTED_STORED:
                rejected_a.append((timestamp, value, group, spec.table))
        else:
            acc_a.value_sum += value
            acc_a.valid_count += 1

        if value < lo_b or value > hi_b:
            acc_b.highres_outlier_count += 1
            n_b += 1
            if hourly:
                n_b_hourly += 1
            if len(rejected_b) < MAX_REJECTED_STORED:
                rejected_b.append((timestamp, value, group, spec.table))
        else:
            acc_b.value_sum += value
            acc_b.valid_count += 1

        # The hourly cloud is kept in full and only the minute-resolution one is
        # subsampled: the filled days are ~0.05 % of the values, so a shared
        # subsampling rate would leave the green points far too sparse to see,
        # and keeping them all costs at most a few tens of thousands of points.
        if (spec.table.strip().lower() == HOURLY_TABLE
                or sample_probability >= 1.0
                or rng.random() < sample_probability):
            sample.append((timestamp, value, group, spec.table))

    return {
        "daily_a": daily_a,
        "daily_b": daily_b,
        "rejected_a": rejected_a,
        "rejected_b": rejected_b,
        "sample": sample,
        "n_rejected_a": n_a,
        "n_rejected_b": n_b,
        "n_rejected_a_hourly": n_a_hourly,
        "n_rejected_b_hourly": n_b_hourly,
        "n_fill_days": len(fill_days),
    }


def daily_band(
    daily: dict[dt.date, DailyAccumulator], multiplier: float
) -> dict[str, float] | None:
    means = np.asarray(
        [acc.value_sum / acc.valid_count for acc in daily.values() if acc.valid_count],
        dtype=np.float64,
    )
    return iqr_limits(means, multiplier) if means.size else None


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def read_column(path: Path, column: str) -> dict[dt.date, float | None]:
    out: dict[dt.date, float | None] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            text = (row.get(column) or "").strip()
            out[dt.date.fromisoformat(row["date"])] = float(text) if text else None
    return out


def compare_series(path_a: Path, path_b: Path, column: str) -> dict[str, object]:
    a = read_column(path_a, column)
    b = read_column(path_b, column)
    diffs: list[float] = []
    only_a = only_b = 0
    for day in sorted(set(a) | set(b)):
        va, vb = a.get(day), b.get(day)
        if va is None and vb is None:
            continue
        if va is None:
            only_b += 1
        elif vb is None:
            only_a += 1
        else:
            diffs.append(vb - va)
    arr = np.asarray(diffs, dtype=np.float64)
    return {
        "column": column,
        "n_days_compared": int(arr.size),
        "n_days_only_in_A": only_a,
        "n_days_only_in_B": only_b,
        "n_days_differing": int(np.count_nonzero(np.abs(arr) > 1e-9)) if arr.size else 0,
        "max_abs_diff": float(np.abs(arr).max()) if arr.size else None,
        "mean_abs_diff": float(np.abs(arr).mean()) if arr.size else None,
        "mean_signed_diff": float(arr.mean()) if arr.size else None,
        "p99_abs_diff": float(np.percentile(np.abs(arr), 99)) if arr.size else None,
    }


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_resolution_compare(
    path: Path,
    station: str,
    band_a: dict[str, float],
    bands_b: dict[str, dict[str, float]],
    spans: dict[str, tuple[dt.date, dt.date]],
    rejected_a: list[tuple[dt.datetime, float, str, str]],
    rejected_b: list[tuple[dt.datetime, float, str, str]],
    sample: list[tuple[dt.datetime, float, str, str]],
) -> None:
    """Raw native-resolution samples vs time, with the IQR-rejected ones crossed.

    Everything is split by NMDB table so the fallback's contribution stays
    visible: ``revori`` samples are grey and rejected with red crosses, the days
    filled from the hourly companion are green and rejected with dark-green
    crosses.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(13.0, 7.8), sharex=True)
    revori_pts = [
        (x, y) for x, y, _g, table in sample
        if table.strip().lower() != HOURLY_TABLE
    ]
    hourly_pts = [
        (x, y) for x, y, _g, table in sample
        if table.strip().lower() == HOURLY_TABLE
    ]

    panels = (
        (axes[0], "A: one pooled band (current pipeline)", rejected_a, None),
        (axes[1], "B: one band per native resolution", rejected_b, bands_b),
    )
    for axis, label, rejected, band_set in panels:
        if revori_pts:
            axis.plot([p[0] for p in revori_pts], [p[1] for p in revori_pts],
                      linestyle="none", marker=".", markersize=0.7,
                      color="0.75", markeredgewidth=0,
                      label=f"revori samples, subsampled (n={len(revori_pts):,})")
        if hourly_pts:
            axis.plot([p[0] for p in hourly_pts], [p[1] for p in hourly_pts],
                      linestyle="none", marker=".", markersize=1.3,
                      color=HOURLY_SAMPLE_COLOR, markeredgewidth=0,
                      label=f"1h samples, filled days, all (n={len(hourly_pts):,})")
        rejected_revori = [
            (p[0], p[1]) for p in rejected
            if p[3].strip().lower() != HOURLY_TABLE
        ]
        rejected_hourly = [
            (p[0], p[1]) for p in rejected
            if p[3].strip().lower() == HOURLY_TABLE
        ]
        if rejected_revori:
            axis.plot([p[0] for p in rejected_revori],
                      [p[1] for p in rejected_revori],
                      linestyle="none", marker="x", markersize=4.2,
                      markeredgewidth=0.9, color="C3",
                      label=f"IQR-rejected revori (n={len(rejected_revori):,})")
        if rejected_hourly:
            axis.plot([p[0] for p in rejected_hourly],
                      [p[1] for p in rejected_hourly],
                      linestyle="none", marker="x", markersize=4.6,
                      markeredgewidth=1.0, color=HOURLY_REJECTED_COLOR,
                      label=f"IQR-rejected 1h (n={len(rejected_hourly):,})")
        if band_set is None:
            axis.axhline(band_a["lower"], color="C0", ls="--", lw=1.0,
                         label=f"band [{band_a['lower']:.2f}, {band_a['upper']:.2f}]")
            axis.axhline(band_a["upper"], color="C0", ls="--", lw=1.0)
        else:
            for group, limits in sorted(band_set.items()):
                lo, hi = spans[group]
                axis.hlines([limits["lower"], limits["upper"]], lo, hi,
                            color="C0", ls="--", lw=1.0)
                axis.annotate(
                    f"{group}: [{limits['lower']:.2f}, {limits['upper']:.2f}]",
                    xy=(lo, limits["upper"]), xytext=(3, 3),
                    textcoords="offset points", fontsize=7.5, color="C0",
                )
        for group, (lo, _hi) in sorted(spans.items()):
            axis.axvline(lo, color="0.35", ls=":", lw=1.0)
        axis.set_title(f"{station} — {label}", fontsize=11, fontweight="bold")
        axis.set_ylabel("count rate (counts/s)")
        axis.tick_params(direction="in", top=True, right=True)
        axis.legend(loc="upper right", fontsize=8, frameon=False, markerscale=3)

    axes[1].set_xlabel("time (UTC)")
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    groups = "   ".join(
        f"{g}: {spans[g][0]}..{spans[g][1]}" for g in sorted(spans)
    )
    figure.suptitle(
        f"{station} — native-resolution IQR, pooled (A) vs per-resolution (B)\n"
        f"resolution groups → {groups}\n"
        f"dots: grey = revori (subsampled), green = 1h fill (complete)   "
        f"crosses: red = rejected revori, dark green = rejected 1h",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def plot_daily_ab(path: Path, station: str, csv_a: Path, csv_b: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    a = read_column(csv_a, "daily_value_candidate")
    b = read_column(csv_b, "daily_value_candidate")
    days = sorted(set(a) | set(b))
    diff = [
        (b[d] - a[d]) if a.get(d) is not None and b.get(d) is not None else np.nan
        for d in days
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(13.0, 6.4), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(days, [a.get(d) for d in days], ".", ms=2.0, color="0.55",
                 label="A: pooled band")
    axes[0].plot(days, [b.get(d) for d in days], "x", ms=3.0, mew=0.7,
                 color="C0", label="B: per-resolution band")
    axes[0].set_ylabel("daily value candidate (counts/s)")
    axes[0].set_title(f"{station} — final daily series, method A vs B",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8, frameon=False, markerscale=3)
    axes[1].plot(days, diff, ".", ms=2.5, color="C3")
    axes[1].axhline(0, color="0.3", lw=0.8)
    axes[1].set_ylabel("B − A (counts/s)")
    axes[1].set_xlabel("time (UTC)")
    for axis in axes:
        axis.tick_params(direction="in", top=True, right=True)
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    figure.tight_layout()
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "TEST: compare pooled-IQR (A) with per-native-resolution-IQR (B) on "
            "stations whose sampling resolution changes over time. Writes only "
            "to *_test paths and leaves the production IQR outputs untouched."
        )
    )
    parser.add_argument(
        "--input-dir", type=Path, action="append",
        help=(
            "directory holding downloaded chunks; repeat the flag to declare a "
            "fallback chain. The first is the preferred table and later ones "
            "only fill the whole days it is missing (default: "
            "data/nmdb_best_revori then data/nmdb_best_1h, i.e. exactly what "
            "the production filter reads)"
        ),
    )
    parser.add_argument("--output-dir", type=Path,
                        default=BASE / "data" / "nmdb_daily_iqr_test")
    parser.add_argument("--stations", type=parse_stations,
                        default=list(DEFAULT_TEST_STATIONS))
    parser.add_argument("--start", type=parse_date, default=DEFAULT_START)
    parser.add_argument("--end", type=parse_date, default=DEFAULT_END)
    parser.add_argument("--iqr-multiplier", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=17,
                        help="seed for the grey display subsample (default: 17)")
    parser.add_argument("--existing-summary", type=Path, default=None,
                        help="optional previous production summary; only its "
                             "per-station band and date range are quoted for "
                             "reference. Never auto-detected, because a stale "
                             "file from a different date range would read as a "
                             "method difference.")
    return parser


def resolve_existing_summary(explicit: Path | None) -> Path | None:
    """The older summary to quote a reference band from -- opt-in only.

    This used to auto-detect a previous production summary (``nmdb_daily_iqr``,
    then ``_bak2``, then ``_bak``). That silently picked whatever file happened
    to exist, and those runs did not all cover the same date range: a band quoted
    from a 2011-2026 run beside this run's 2001-2026 band reads as a difference
    in method when it is mostly a difference in range. So the reference is now
    always named explicitly by the caller.
    """
    return explicit


def previous_bands(path: Path) -> dict[str, dict[str, object]]:
    """Bands from the existing production run, to separate the two effects."""
    import json as _json

    if not path.is_file():
        return {}
    try:
        payload = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, dict[str, object]] = {}
    for entry in payload.get("stations", []):
        station = entry.get("station")
        if station:
            out[station] = {
                "range": payload.get("date_range"),
                "highres_iqr": entry.get("highres_iqr"),
                "highres_outliers": entry.get("highres_outliers"),
            }
    return out


def main() -> int:
    args = build_parser().parse_args()
    if args.start > args.end:
        raise SystemExit("--start must not be after --end")
    if not args.input_dir:
        args.input_dir = [
            BASE / "data" / "nmdb_best_revori",
            BASE / "data" / "nmdb_best_1h",
        ]
    for directory in args.input_dir:
        if not directory.is_dir():
            raise SystemExit(f"input dir does not exist: {directory}")

    data_dir = args.output_dir / "data"
    plot_dir = args.output_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary_source = resolve_existing_summary(args.existing_summary)
    old = previous_bands(summary_source) if summary_source else {}
    if summary_source:
        print(f"reference band source: {summary_source if old else '(no band in file)'}")

    reports: list[dict[str, object]] = []
    for index, station in enumerate(args.stations, start=1):
        print(f"[{index}/{len(args.stations)}] {station}", flush=True)
        usable_sources, specs, skipped = load_sources(
            args.input_dir, station, args.start, args.end
        )
        if not usable_sources:
            print("  no usable files")
            reports.append({"station": station, "status": "no_files",
                            "skipped_files": skipped})
            continue

        spans = group_spans(specs)
        meta = group_meta(specs)
        per_group = group_values(usable_sources, specs, args.start, args.end)
        if not per_group:
            print("  no usable values")
            reports.append({"station": station, "status": "no_values",
                            "skipped_files": skipped})
            continue
        pooled = np.concatenate(list(per_group.values()))
        band_a = iqr_limits(pooled, args.iqr_multiplier)
        bands_b = {g: iqr_limits(v, args.iqr_multiplier) for g, v in per_group.items()}

        total_points = int(pooled.size)
        del pooled
        probability = min(1.0, DISPLAY_POINTS / max(total_points, 1))
        result = accumulate_both(usable_sources, specs, args.start, args.end,
                                 band_a, bands_b, probability, rng)
        daily_a = result["daily_a"]
        daily_b = result["daily_b"]
        limits_a = daily_band(daily_a, args.iqr_multiplier)
        limits_b = daily_band(daily_b, args.iqr_multiplier)

        csv_a = data_dir / f"{station}_daily_iqr_A_test.csv"
        csv_b = data_dir / f"{station}_daily_iqr_B_test.csv"
        days_a = write_daily_csv(csv_a, args.start, args.end, daily_a, limits_a)
        days_b = write_daily_csv(csv_b, args.start, args.end, daily_b, limits_b)

        plot_resolution_compare(
            plot_dir / f"{station}_iqr_resolution_compare_test.pdf",
            station, band_a, bands_b, spans,
            result["rejected_a"], result["rejected_b"], result["sample"],
        )
        plot_daily_ab(plot_dir / f"{station}_daily_AB_test.pdf", station, csv_a, csv_b)

        report = {
            "station": station,
            "status": "processed",
            "n_native_values": total_points,
            "groups": {
                g: {
                    "n_values": int(per_group[g].size),
                    "span": [spans[g][0].isoformat(), spans[g][1].isoformat()],
                    "band_B": bands_b[g],
                    # raw ORIGINAL RES header strings seen in this group, so a
                    # cosmetic label (e.g. AATB's "multiple: ...") is visible
                    "raw_headers": meta.get(g, {}).get("raw_headers", []),
                    "nmdb_tables": meta.get(g, {}).get("nmdb_tables", []),
                }
                for g in sorted(per_group)
            },
            "band_A": band_a,
            "previous_production_band": old.get(station, {}).get("highres_iqr"),
            "previous_production_range": old.get(station, {}).get("range"),
            "highres_rejected_A": result["n_rejected_a"],
            "highres_rejected_B": result["n_rejected_b"],
            "highres_rejected_A_hourly": result["n_rejected_a_hourly"],
            "highres_rejected_B_hourly": result["n_rejected_b_hourly"],
            "days_filled_from_fallback": result["n_fill_days"],
            "daily_iqr_A": limits_a,
            "daily_iqr_B": limits_b,
            "daily_counts_A": days_a,
            "daily_counts_B": days_b,
            "diff_daily_mean_after_highres_iqr": compare_series(
                csv_a, csv_b, "daily_mean_after_highres_iqr"),
            "diff_daily_value_candidate": compare_series(
                csv_a, csv_b, "daily_value_candidate"),
            "skipped_files": skipped,
            "outputs": {"csv_A": str(csv_a), "csv_B": str(csv_b)},
        }
        reports.append(report)

        print(f"  groups ({len(per_group)}): " + "; ".join(
            f"{g} n={per_group[g].size:,} "
            f"[{spans[g][0]}..{spans[g][1]}] "
            f"band=[{bands_b[g]['lower']:.2f},{bands_b[g]['upper']:.2f}]"
            for g in sorted(per_group)))
        if len(per_group) == 1:
            print("  -> single resolution group: A and B must be identical")
        print(f"  days filled from a fallback directory: {result['n_fill_days']}")
        prev = old.get(station, {}).get("highres_iqr")
        if prev:
            span = old.get(station, {}).get("range") or []
            label = "..".join(str(x) for x in span) if span else "reference"
            print(f"  reference band ({label}): "
                  f"[{prev['lower']:.2f}, {prev['upper']:.2f}]")
        print(f"  band A (this range):          [{band_a['lower']:.2f}, {band_a['upper']:.2f}]")
        print(f"  rejected sub-daily values: "
              f"A={result['n_rejected_a']:,} "
              f"(of which 1h: {result['n_rejected_a_hourly']:,})   "
              f"B={result['n_rejected_b']:,} "
              f"(of which 1h: {result['n_rejected_b_hourly']:,})")
        for key in ("diff_daily_mean_after_highres_iqr", "diff_daily_value_candidate"):
            d = report[key]
            print(f"  {d['column']}: {d['n_days_differing']} days differ, "
                  f"max|d|={d['max_abs_diff']}, mean|d|={d['mean_abs_diff']}, "
                  f"only_A={d['n_days_only_in_A']}, only_B={d['n_days_only_in_B']}")

    summary_path = args.output_dir / "summary_test.json"
    summary_path.write_text(
        json.dumps({
            "note": "TEST comparison script; production outputs untouched",
            "date_range": [args.start.isoformat(), args.end.isoformat()],
            "iqr_multiplier": args.iqr_multiplier,
            "methods": {
                "A": "one pooled 3*IQR band over the whole range (current pipeline)",
                "B": "one 3*IQR band per native resolution, merged into daily means",
            },
            "stations": reports,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
