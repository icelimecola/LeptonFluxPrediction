#!/usr/bin/env python3
"""Audit downloaded NMDB chunks without loading full station histories."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from download_nmdb import (
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_STATIONS,
    latest_complete_chunk_end,
    parse_date,
    parse_duration_minutes,
    parse_stations,
)
from station_metadata import nmdb_table_name


FILE_RE = re.compile(
    r"^(?P<station>[A-Z0-9]{4,5})_"
    r"(?P<start>\d{8})_(?P<end>\d{8})_"
    r"(?P<resolution>best|\d+min)\.txt"
    r"(?P<no_data>\.no_data)?$"
)
NULL_VALUES = {"", "null", "nan", "na", "none"}


@dataclass
class FileAudit:
    path: str
    kind: str
    station: str
    requested_start: str
    requested_end: str
    requested_resolution: str
    status: str
    issues: str
    nmdb_table: str = ""
    header_start: str = ""
    header_end: str = ""
    original_resolution: str = ""
    effective_resolution_minutes: int | str = ""
    averaging: str = ""
    rows: int = 0
    valid_values: int = 0
    null_values: int = 0
    malformed_rows: int = 0
    duplicate_timestamps: int = 0
    nonmonotonic_timestamps: int = 0
    missing_timestamp_slots: int = 0
    irregular_steps: int = 0
    expected_rows: int | str = ""
    first_timestamp: str = ""
    last_timestamp: str = ""


def parse_filename(path: Path) -> tuple[str, dt.date, dt.date, str, bool] | None:
    match = FILE_RE.fullmatch(path.name)
    if not match:
        return None
    return (
        match.group("station"),
        dt.datetime.strptime(match.group("start"), "%Y%m%d").date(),
        dt.datetime.strptime(match.group("end"), "%Y%m%d").date(),
        match.group("resolution"),
        bool(match.group("no_data")),
    )


def parse_summary_line(line: str) -> tuple[str, str] | None:
    match = re.match(r"^#\s*([^:]+):\s*(.*?)\s*$", line)
    if not match:
        return None
    return match.group(1).strip().upper(), match.group(2).strip()


def parse_header_datetime(value: str) -> dt.datetime | None:
    value = re.sub(r"\s+UTC\s*$", "", value.strip())
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


def effective_resolution(summary: dict[str, str]) -> int | None:
    averaging = summary.get("AVERAGING", "").strip()
    if averaging.lower() == "no":
        return parse_duration_minutes(summary.get("ORIGINAL RES", ""))
    if "/" in averaging:
        return parse_duration_minutes(averaging.split("/", 1)[1])
    return None


def expected_resolution(requested: str) -> int | None:
    if requested == "best":
        return None
    return int(requested.removesuffix("min"))


def infer_resolution_minutes(step_counts: Counter[int]) -> int | None:
    minute_steps = {
        seconds: count
        for seconds, count in step_counts.items()
        if seconds > 0 and seconds % 60 == 0
    }
    if not minute_steps:
        return None
    nominal_seconds = min(
        minute_steps,
        key=lambda seconds: (-minute_steps[seconds], seconds),
    )
    return nominal_seconds // 60


def scan_data_file(
    path: Path,
    station: str,
    requested_start: dt.date,
    requested_end: dt.date,
    requested_resolution: str,
) -> FileAudit:
    summary: dict[str, str] = {}
    rows = 0
    valid_values = 0
    null_values = 0
    malformed_rows = 0
    duplicate_timestamps = 0
    nonmonotonic_timestamps = 0
    missing_timestamp_slots = 0
    irregular_steps = 0
    first_timestamp: dt.datetime | None = None
    last_timestamp: dt.datetime | None = None
    previous_timestamp: dt.datetime | None = None
    resolution_minutes: int | None = None
    unresolved_step_counts: Counter[int] = Counter()

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                parsed = parse_summary_line(line)
                if parsed:
                    summary[parsed[0]] = parsed[1]
                continue
            if line.lower().startswith("start_date_time"):
                resolution_minutes = effective_resolution(summary)
                continue

            rows += 1
            fields = line.split(";", 1)
            if len(fields) != 2:
                malformed_rows += 1
                continue
            try:
                timestamp = dt.datetime.fromisoformat(fields[0].strip())
            except ValueError:
                malformed_rows += 1
                continue

            value_text = fields[1].strip().lower()
            if value_text in NULL_VALUES:
                null_values += 1
            else:
                try:
                    value = float(value_text)
                except ValueError:
                    malformed_rows += 1
                    continue
                if math.isfinite(value):
                    valid_values += 1
                else:
                    null_values += 1

            if first_timestamp is None:
                first_timestamp = timestamp
            if previous_timestamp is not None:
                step_seconds = int((timestamp - previous_timestamp).total_seconds())
                if step_seconds == 0:
                    duplicate_timestamps += 1
                elif step_seconds < 0:
                    nonmonotonic_timestamps += 1
                elif resolution_minutes is not None:
                    expected_seconds = resolution_minutes * 60
                    if step_seconds > expected_seconds:
                        if step_seconds % expected_seconds == 0:
                            missing_timestamp_slots += step_seconds // expected_seconds - 1
                        else:
                            irregular_steps += 1
                    elif step_seconds != expected_seconds:
                        irregular_steps += 1
                else:
                    unresolved_step_counts[step_seconds] += 1
            previous_timestamp = timestamp
            last_timestamp = timestamp

    issues: list[str] = []
    fatal = False
    header_station = summary.get("STATION", "")
    if header_station != station:
        issues.append(f"header station={header_station or '<missing>'}")
        fatal = True
    expected_table = nmdb_table_name(station)
    if summary.get("NMDB TABLE", "").lower() != expected_table.lower():
        issues.append(
            f"table={summary.get('NMDB TABLE', '<missing>')}, "
            f"expected={expected_table}"
        )
        fatal = True
    if "corr_for_efficiency" not in summary.get("DATA TYPE", "").lower():
        issues.append(f"data type={summary.get('DATA TYPE', '<missing>')}")
        fatal = True

    resolution_minutes = effective_resolution(summary)
    if resolution_minutes is None and requested_resolution == "best":
        resolution_minutes = infer_resolution_minutes(unresolved_step_counts)
        if resolution_minutes is not None:
            expected_seconds = resolution_minutes * 60
            for step_seconds, count in unresolved_step_counts.items():
                if step_seconds > expected_seconds:
                    if step_seconds % expected_seconds == 0:
                        missing_timestamp_slots += count * (
                            step_seconds // expected_seconds - 1
                        )
                    else:
                        irregular_steps += count
                elif step_seconds != expected_seconds:
                    irregular_steps += count
    if resolution_minutes is None:
        issues.append("invalid effective resolution")
        fatal = True
    if requested_resolution == "best":
        if summary.get("AVERAGING", "").strip().lower() != "no":
            issues.append(f"best request averaged={summary.get('AVERAGING', '<missing>')}")
            fatal = True
    else:
        target = expected_resolution(requested_resolution)
        if resolution_minutes != target:
            issues.append(f"effective resolution={resolution_minutes}, expected={target}")
            fatal = True

    header_start = parse_header_datetime(summary.get("START TIME", ""))
    header_end = parse_header_datetime(summary.get("END TIME", ""))
    if header_start is None or header_start.date() != requested_start:
        issues.append("header start does not match filename")
        fatal = True
    if header_end is None or header_end.date() != requested_end:
        issues.append("header end does not match filename")
        fatal = True
    if malformed_rows:
        issues.append(f"malformed rows={malformed_rows}")
        fatal = True
    if duplicate_timestamps:
        issues.append(f"duplicate timestamps={duplicate_timestamps}")
        fatal = True
    if nonmonotonic_timestamps:
        issues.append(f"nonmonotonic timestamps={nonmonotonic_timestamps}")
        fatal = True
    if missing_timestamp_slots:
        issues.append(f"missing timestamp slots={missing_timestamp_slots}")
    if irregular_steps:
        issues.append(f"irregular steps={irregular_steps}")
    if null_values:
        issues.append(f"null values={null_values}")

    expected_rows: int | str = ""
    if header_start is not None and header_end is not None and resolution_minutes:
        duration_seconds = int((header_end - header_start).total_seconds())
        expected_rows = duration_seconds // (resolution_minutes * 60) + 1
        if rows != expected_rows:
            issues.append(f"rows={rows}, expected={expected_rows}")

    if fatal:
        status = "invalid"
    elif missing_timestamp_slots or irregular_steps or rows != expected_rows:
        status = "valid_with_gaps"
    else:
        status = "valid"

    return FileAudit(
        path=str(path),
        kind="data",
        station=station,
        requested_start=requested_start.isoformat(),
        requested_end=requested_end.isoformat(),
        requested_resolution=requested_resolution,
        status=status,
        issues="; ".join(issues),
        nmdb_table=summary.get("NMDB TABLE", ""),
        header_start=header_start.isoformat(sep=" ") if header_start else "",
        header_end=header_end.isoformat(sep=" ") if header_end else "",
        original_resolution=summary.get("ORIGINAL RES", ""),
        effective_resolution_minutes=resolution_minutes or "",
        averaging=summary.get("AVERAGING", ""),
        rows=rows,
        valid_values=valid_values,
        null_values=null_values,
        malformed_rows=malformed_rows,
        duplicate_timestamps=duplicate_timestamps,
        nonmonotonic_timestamps=nonmonotonic_timestamps,
        missing_timestamp_slots=missing_timestamp_slots,
        irregular_steps=irregular_steps,
        expected_rows=expected_rows,
        first_timestamp=first_timestamp.isoformat(sep=" ") if first_timestamp else "",
        last_timestamp=last_timestamp.isoformat(sep=" ") if last_timestamp else "",
    )


def audit_no_data_marker(
    path: Path,
    station: str,
    requested_start: dt.date,
    requested_end: dt.date,
    requested_resolution: str,
) -> FileAudit:
    text = path.read_text(encoding="utf-8", errors="replace")
    issues: list[str] = []
    if "NMDB no-data marker" not in text:
        issues.append("unrecognized no-data marker")
    summary: dict[str, str] = {}
    for line in text.splitlines():
        parsed = parse_summary_line(line)
        if parsed:
            summary[parsed[0]] = parsed[1]
    marker_table = summary.get("NMDB TABLE", "")
    expected_table = nmdb_table_name(station)
    if marker_table:
        if marker_table.lower() != expected_table.lower():
            issues.append(f"table={marker_table}, expected={expected_table}")
    elif expected_table.lower() != "revised original":
        issues.append(f"missing NMDB table; expected={expected_table}")
    status = "no_data" if not issues else "invalid"
    return FileAudit(
        path=str(path),
        kind="no_data",
        station=station,
        requested_start=requested_start.isoformat(),
        requested_end=requested_end.isoformat(),
        requested_resolution=requested_resolution,
        status=status,
        issues="; ".join(issues),
        nmdb_table=marker_table,
    )


def add_months(day: dt.date, months: int) -> dt.date:
    index = day.year * 12 + day.month - 1 + months
    return dt.date(index // 12, index % 12 + 1, 1)


def iter_months(start: dt.date, end: dt.date):
    current = start.replace(day=1)
    while current <= end:
        month_end = min(end, add_months(current, 1) - dt.timedelta(days=1))
        yield max(start, current), month_end
        current = add_months(current, 1)


def intervals_cover(
    intervals: list[tuple[dt.date, dt.date]], start: dt.date, end: dt.date
) -> bool:
    cursor = start
    for interval_start, interval_end in sorted(intervals):
        if interval_end < cursor:
            continue
        if interval_start > cursor:
            return False
        cursor = max(cursor, interval_end + dt.timedelta(days=1))
        if cursor > end:
            return True
    return cursor > end


def month_status(
    audits: list[FileAudit], station: str, start: dt.date, end: dt.date
) -> tuple[str, int, int]:
    data_intervals: list[tuple[dt.date, dt.date]] = []
    no_data_intervals: list[tuple[dt.date, dt.date]] = []
    covering_files = 0
    invalid_files = 0
    for audit in audits:
        if audit.station != station:
            continue
        audit_start = dt.date.fromisoformat(audit.requested_start)
        audit_end = dt.date.fromisoformat(audit.requested_end)
        if audit_end < start or audit_start > end:
            continue
        covering_files += 1
        interval = (max(start, audit_start), min(end, audit_end))
        if audit.status == "invalid":
            invalid_files += 1
        elif audit.kind == "data":
            data_intervals.append(interval)
        elif audit.kind == "no_data":
            no_data_intervals.append(interval)

    if intervals_cover(data_intervals, start, end):
        status = "data"
    elif intervals_cover(no_data_intervals, start, end):
        status = "no_data"
    elif intervals_cover(data_intervals + no_data_intervals, start, end):
        status = "mixed"
    elif data_intervals or no_data_intervals:
        status = "partial"
    elif invalid_files:
        status = "invalid"
    else:
        status = "missing"
    return status, covering_files, invalid_files


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Audit NMDB files, values, timestamps, and monthly coverage."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=base / "rawdata" / "nmdb_best"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=base / "rawdata" / "nmdb_audit"
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
        help="audit through the latest complete --chunk-months block",
    )
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=3,
        help="months per download block when using --latest (default: 3)",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="return a nonzero status if any month is missing, partial, or invalid",
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
    if not args.input_dir.is_dir():
        raise SystemExit(f"input directory does not exist: {args.input_dir}")

    candidates = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_file() and (path.name.endswith(".txt") or path.name.endswith(".no_data"))
    )
    audits: list[FileAudit] = []
    ignored_files: list[str] = []
    for index, path in enumerate(candidates, start=1):
        parsed = parse_filename(path)
        if parsed is None:
            ignored_files.append(str(path))
            continue
        station, requested_start, requested_end, resolution, is_no_data = parsed
        print(f"[{index}/{len(candidates)}] audit {path.name}")
        if is_no_data:
            audit = audit_no_data_marker(
                path, station, requested_start, requested_end, resolution
            )
        else:
            audit = scan_data_file(
                path, station, requested_start, requested_end, resolution
            )
        audits.append(audit)
        print(
            f"  {audit.status}: rows={audit.rows}, valid={audit.valid_values}, "
            f"null={audit.null_values}"
        )

    file_rows = [asdict(audit) for audit in audits]
    file_fields = [field.name for field in FileAudit.__dataclass_fields__.values()]
    write_csv(args.output_dir / "files.csv", file_rows, file_fields)

    month_rows: list[dict] = []
    status_counts: dict[str, int] = {}
    for station in args.stations:
        for month_start, month_end in iter_months(args.start, args.end):
            status, covering_files, invalid_files = month_status(
                audits, station, month_start, month_end
            )
            status_counts[status] = status_counts.get(status, 0) + 1
            month_rows.append(
                {
                    "station": station,
                    "month": month_start.strftime("%Y-%m"),
                    "start": month_start.isoformat(),
                    "end": month_end.isoformat(),
                    "status": status,
                    "covering_files": covering_files,
                    "invalid_files": invalid_files,
                }
            )
    month_fields = [
        "station",
        "month",
        "start",
        "end",
        "status",
        "covering_files",
        "invalid_files",
    ]
    write_csv(args.output_dir / "months.csv", month_rows, month_fields)

    summary = {
        "input_dir": str(args.input_dir),
        "date_range": [args.start.isoformat(), args.end.isoformat()],
        "stations": args.stations,
        "files_scanned": len(audits),
        "ignored_files": ignored_files,
        "file_status_counts": {
            status: sum(audit.status == status for audit in audits)
            for status in sorted({audit.status for audit in audits})
        },
        "month_status_counts": status_counts,
        "total_rows": sum(audit.rows for audit in audits),
        "total_valid_values": sum(audit.valid_values for audit in audits),
        "total_null_values": sum(audit.null_values for audit in audits),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    print(f"Reports: {args.output_dir}")
    print(f"Files: {summary['file_status_counts']}")
    print(f"Months: {status_counts}")
    incomplete = sum(
        status_counts.get(status, 0) for status in ("missing", "partial", "invalid")
    )
    return 1 if args.fail_on_missing and incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
