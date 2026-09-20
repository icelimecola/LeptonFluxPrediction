#!/usr/bin/env python3
"""TEST SCRIPT -- compare two IQR strategies on mixed-resolution stations.

This file is deliberately standalone: it does NOT modify ``nmdb_filter_iqr.py``
and it only ever writes to ``*_test`` paths.

Method A -- the current pipeline
    Pool every sub-daily value of the station over the whole ``[start, end]``
    range into one array, compute a single ``3*IQR`` band, and drop every value
    outside it.

Method B -- resolution-aware
    Group the sub-daily values by their native resolution (each file's
    ``ORIGINAL RES`` header), compute one ``3*IQR`` band *per group*, and drop a
    value using its own group's band. Surviving values are then merged into daily
    means exactly as in method A.

    The daily-level (second) IQR stage is still pooled in both methods: every
    daily mean belongs to one resolution group anyway, and keeping that stage
    identical isolates the effect being measured.

Motivation
    The 2001-2010 backfill added files whose native resolution differs from the
    station's modern files: JUNG / JUNG1 are 60 min for 2001-2008 and 1 min from
    2008-08; INVK and THUL are 60 min for part of 2001; MXCO is 5 min before
    2008. Note that for MXCO both eras come from the SAME NMDB table
    ("revised original"): the table choice does not determine the sampling
    resolution, the station's archive history does. A 5-minute sample and a
    1-minute sample do not have the same spread, so one pooled band is a
    compromise that fits neither era. This script measures how much that
    actually matters, and plots the raw-resolution series with the rejected
    samples marked as red crosses.

Outputs (all suffixed ``_test``, under ``rawdata/nmdb_daily_iqr_test``)
    data/<ST>_daily_iqr_A_test.csv     method A daily series (production columns)
    data/<ST>_daily_iqr_B_test.csv     method B daily series
    plots/<ST>_iqr_resolution_compare_test.pdf
                                       raw sub-daily samples vs time with the
                                       resolution groups, each method's band, and
                                       the rejected samples as red crosses
    plots/<ST>_daily_AB_test.pdf       daily series A vs B plus their difference
    summary_test.json                  bands, rejection counts, day-level diffs

NOT RUN YET -- written but not executed, per instruction.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

from audit_nmdb import parse_filename, scan_data_file
from nmdb_download import parse_date, parse_stations
from nmdb_filter_iqr import (
    DailyAccumulator,
    parse_accept_tables,
    discover_station_files,
    iqr_limits,
    iter_data_rows,
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
# AATB is included as a control: its header says "multiple: min = 0 min,
# max = 1 min" on two files, but the sampling is a uniform 60 s, so it must end
# up as ONE group and A must equal B exactly.
DEFAULT_TEST_STATIONS = ["JUNG", "JUNG1", "INVK", "THUL", "MXCO", "AATB"]
DISPLAY_POINTS = 150_000      # cap for the grey background cloud
MAX_REJECTED_STORED = 400_000  # safety cap on the red crosses


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #
def load_units(
    input_dir: Path,
    station: str,
    start: dt.date,
    end: dt.date,
    accept_tables: list[str] | None = None,
) -> tuple[list[tuple[Path, str, str, str]], list[dict[str, str]]]:
    """Return ``(path, resolution_group, nmdb_table, raw_header)`` units.

    The grouping key is the audit's normalised ``effective_resolution_minutes``,
    NOT the raw ``ORIGINAL RES`` header string. The header is unreliable for
    grouping: AATB writes ``multiple: min = 0 min, max = 1 min`` on some files
    whose sampling is in fact a uniform 60 s, which would otherwise create
    spurious resolution groups. The raw header is kept for reporting only.
    """
    units: list[tuple[Path, str, str, str]] = []
    skipped: list[dict[str, str]] = []
    for path in discover_station_files(input_dir, station, start, end):
        parsed = parse_filename(path)
        assert parsed is not None
        _, file_start, file_end, resolution, _ = parsed
        audit = scan_data_file(path, station, file_start, file_end, resolution,
                               accept_tables=accept_tables)
        if audit.status == "invalid":
            skipped.append({"path": str(path), "reason": audit.issues})
            continue
        group = f"{audit.effective_resolution_minutes} min"
        units.append((path, group, audit.nmdb_table.strip(),
                      audit.original_resolution.strip() or "unknown"))
    validate_nonoverlap([path for path, *_ in units])
    return units, skipped


def group_values(
    units: list[tuple[Path, str, str, str]], start: dt.date, end: dt.date
) -> dict[str, np.ndarray]:
    """Pool the valid sub-daily values of each native-resolution group."""
    blocks: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    for path, group, _table, _raw in units:
        values: list[float] = []
        for timestamp, value in iter_data_rows(path):
            if value is None:
                continue
            if start <= timestamp.date() <= end:
                values.append(value)
        if values:
            blocks[group].append(np.asarray(values, dtype=np.float64))
    return {group: np.concatenate(parts) for group, parts in blocks.items()}


def group_spans(
    units: list[tuple[Path, str, str, str]]
) -> dict[str, tuple[dt.date, dt.date]]:
    spans: dict[str, list[dt.date]] = collections.defaultdict(list)
    for path, group, _table, _raw in units:
        parsed = parse_filename(path)
        assert parsed is not None
        spans[group].extend([parsed[1], parsed[2]])
    return {group: (min(v), max(v)) for group, v in spans.items()}


def accumulate_both(
    units: list[tuple[Path, str, str, str]],
    start: dt.date,
    end: dt.date,
    band_a: dict[str, float],
    bands_b: dict[str, dict[str, float]],
    sample_probability: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    """One pass over the data applying both methods simultaneously.

    Also collects the ``(timestamp, value, group)`` of every rejected sample for
    the plots, and a Bernoulli subsample of all samples for the grey cloud.
    """
    daily_a: dict[dt.date, DailyAccumulator] = {}
    daily_b: dict[dt.date, DailyAccumulator] = {}
    rejected_a: list[tuple[dt.datetime, float, str]] = []
    rejected_b: list[tuple[dt.datetime, float, str]] = []
    sample: list[tuple[dt.datetime, float]] = []
    n_a = n_b = 0

    for path, group, _table, _raw in units:
        limits_b = bands_b[group]
        lo_a, hi_a = band_a["lower"], band_a["upper"]
        lo_b, hi_b = limits_b["lower"], limits_b["upper"]
        for timestamp, value in iter_data_rows(path):
            day = timestamp.date()
            if day < start or day > end:
                continue
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
                if len(rejected_a) < MAX_REJECTED_STORED:
                    rejected_a.append((timestamp, value, group))
            else:
                acc_a.value_sum += value
                acc_a.valid_count += 1

            if value < lo_b or value > hi_b:
                acc_b.highres_outlier_count += 1
                n_b += 1
                if len(rejected_b) < MAX_REJECTED_STORED:
                    rejected_b.append((timestamp, value, group))
            else:
                acc_b.value_sum += value
                acc_b.valid_count += 1

            if sample_probability >= 1.0 or rng.random() < sample_probability:
                sample.append((timestamp, value))

    return {
        "daily_a": daily_a,
        "daily_b": daily_b,
        "rejected_a": rejected_a,
        "rejected_b": rejected_b,
        "sample": sample,
        "n_rejected_a": n_a,
        "n_rejected_b": n_b,
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
    rejected_a: list[tuple[dt.datetime, float, str]],
    rejected_b: list[tuple[dt.datetime, float, str]],
    sample: list[tuple[dt.datetime, float]],
) -> None:
    """Raw native-resolution samples vs time; red crosses = IQR-rejected."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(13.0, 7.8), sharex=True)
    sx = [p[0] for p in sample]
    sy = [p[1] for p in sample]

    panels = (
        (axes[0], "A: one pooled band (current pipeline)", rejected_a, None),
        (axes[1], "B: one band per native resolution", rejected_b, bands_b),
    )
    for axis, label, rejected, band_set in panels:
        if sx:
            axis.plot(sx, sy, linestyle="none", marker=".", markersize=0.7,
                      color="0.75", markeredgewidth=0,
                      label=f"raw sub-daily samples (subsampled n={len(sx):,})")
        if rejected:
            axis.plot([p[0] for p in rejected], [p[1] for p in rejected],
                      linestyle="none", marker="x", markersize=4.2,
                      markeredgewidth=0.9, color="C3",
                      label=f"IQR-rejected (n={len(rejected):,})")
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
        f"resolution groups → {groups}",
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
    parser.add_argument("--input-dir", type=Path,
                        default=BASE / "rawdata" / "nmdb_best")
    parser.add_argument("--output-dir", type=Path,
                        default=BASE / "rawdata" / "nmdb_daily_iqr_test")
    parser.add_argument("--stations", type=parse_stations,
                        default=list(DEFAULT_TEST_STATIONS))
    parser.add_argument("--start", type=parse_date, default=DEFAULT_START)
    parser.add_argument("--end", type=parse_date, default=DEFAULT_END)
    parser.add_argument("--iqr-multiplier", type=float, default=3.0)
    parser.add_argument(
        "--accept-table",
        type=parse_accept_tables,
        default=parse_accept_tables("revori,1h"),
        metavar="ori|revori|1h[,..]",
        help=(
            "NMDB tables accepted by the per-file audit. Default 'revori,1h' "
            "because this script exists precisely to handle stations that span "
            "two tables; pass just 'revori' to reproduce the production filter, "
            "which then collapses those stations to a single resolution group."
        ),
    )
    parser.add_argument("--seed", type=int, default=17,
                        help="seed for the grey display subsample (default: 17)")
    parser.add_argument("--existing-summary", type=Path, default=None,
                        help="previous production summary, used only to print the "
                             "old band for reference. Default: the first existing "
                             "of rawdata/nmdb_daily_iqr{, _bak2, _bak}")
    return parser


def resolve_existing_summary(explicit: Path | None) -> Path:
    """Pick the production summary to quote the old band from."""
    if explicit is not None:
        return explicit
    for candidate in ("nmdb_daily_iqr", "nmdb_daily_iqr_bak2", "nmdb_daily_iqr_bak"):
        path = BASE / "rawdata" / candidate / "preprocess_summary.json"
        if path.is_file():
            return path
    return BASE / "rawdata" / "nmdb_daily_iqr" / "preprocess_summary.json"


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
    if not args.input_dir.is_dir():
        raise SystemExit(f"input dir does not exist: {args.input_dir}")

    data_dir = args.output_dir / "data"
    plot_dir = args.output_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary_source = resolve_existing_summary(args.existing_summary)
    old = previous_bands(summary_source)
    print(f"reference band source: {summary_source if old else '(none found)'}")

    reports: list[dict[str, object]] = []
    for index, station in enumerate(args.stations, start=1):
        print(f"[{index}/{len(args.stations)}] {station}", flush=True)
        units, skipped = load_units(args.input_dir, station, args.start, args.end,
                                    args.accept_table)
        if not units:
            print("  no usable files")
            reports.append({"station": station, "status": "no_files",
                            "skipped_files": skipped})
            continue

        spans = group_spans(units)
        per_group = group_values(units, args.start, args.end)
        pooled = np.concatenate(list(per_group.values()))
        band_a = iqr_limits(pooled, args.iqr_multiplier)
        bands_b = {g: iqr_limits(v, args.iqr_multiplier) for g, v in per_group.items()}

        total_points = int(pooled.size)
        del pooled
        probability = min(1.0, DISPLAY_POINTS / max(total_points, 1))
        result = accumulate_both(units, args.start, args.end, band_a, bands_b,
                                 probability, rng)
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
                    "raw_headers": sorted({raw for _p, grp, _t, raw in units
                                           if grp == g}),
                    "nmdb_tables": sorted({tab for _p, grp, tab, _r in units
                                           if grp == g}),
                }
                for g in sorted(per_group)
            },
            "band_A": band_a,
            "previous_production_band": old.get(station, {}).get("highres_iqr"),
            "previous_production_range": old.get(station, {}).get("range"),
            "highres_rejected_A": result["n_rejected_a"],
            "highres_rejected_B": result["n_rejected_b"],
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
        prev = old.get(station, {}).get("highres_iqr")
        if prev:
            print(f"  band (production 2011-2026): [{prev['lower']:.2f}, {prev['upper']:.2f}]")
        print(f"  band A (this range):          [{band_a['lower']:.2f}, {band_a['upper']:.2f}]")
        print(f"  rejected sub-daily values: A={result['n_rejected_a']:,}  "
              f"B={result['n_rejected_b']:,}")
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
