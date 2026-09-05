#!/usr/bin/env python3
"""Download corrected neutron-monitor data from NMDB at validated resolution.

The default ``best`` mode keeps each station's finest available native
resolution. The script deliberately downloads small time chunks because NEST
may silently lower the requested resolution for large queries.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from station_metadata import nmdb_table_choice, nmdb_table_name


NMDB_URL = "https://www.nmdb.eu/nest/draw_graph.php"
DEFAULT_START = dt.date(2011, 1, 1)
DEFAULT_END = dt.date(2025, 12, 31)
DEFAULT_STATIONS = (
    "AATB", "APTY", "FSMT", "INVK", "JUNG", "JUNG1",
    "LMKS", "MXCO", "NAIN", "NEWK", "OULU", "PSNM",
    "PWNK", "SOPB", "SOPO", "TERA", "THUL", "YKTK",
)
RESOLUTIONS = {
    "best": 0,
    "2": 2,
    "5": 5,
    "10": 10,
    "30": 30,
    "60": 60,
    "120": 120,
    "180": 180,
    "360": 360,
    "720": 720,
    "1440": 1440,
}


class NMDBError(RuntimeError):
    """Raised when NMDB returns an invalid or unexpected result."""


class NMDBNoData(NMDBError):
    """Raised when NMDB explicitly reports no data for a requested chunk."""


class NMDBRequestError(NMDBError):
    """Raised for transient HTTP, TLS, timeout, or other transport failures."""


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid date {value!r}; expected YYYY-MM-DD"
        ) from exc


def add_months(day: dt.date, months: int) -> dt.date:
    """Return the first day that is ``months`` months after ``day``."""
    month_index = day.year * 12 + day.month - 1 + months
    return dt.date(month_index // 12, month_index % 12 + 1, 1)


def latest_complete_chunk_end(
    start: dt.date,
    chunk_months: int,
    today: dt.date | None = None,
) -> dt.date:
    """Return the end of the latest complete month-based chunk from ``start``."""
    if start.day != 1:
        raise ValueError("start date must be the first day of a month")
    if chunk_months < 1:
        raise ValueError("chunk_months must be at least 1")

    current = today or dt.datetime.now(dt.timezone.utc).date()
    next_start = start
    completed_end = start - dt.timedelta(days=1)
    while add_months(next_start, chunk_months) <= current:
        next_start = add_months(next_start, chunk_months)
        completed_end = next_start - dt.timedelta(days=1)
    return completed_end


def iter_chunks(
    start: dt.date, end: dt.date, chunk_months: int
) -> list[tuple[dt.date, dt.date]]:
    if start > end:
        raise ValueError("start date must not be after end date")
    if chunk_months < 1:
        raise ValueError("chunk_months must be at least 1")

    chunks: list[tuple[dt.date, dt.date]] = []
    chunk_start = start
    while chunk_start <= end:
        next_start = add_months(chunk_start, chunk_months)
        chunk_end = min(end, next_start - dt.timedelta(days=1))
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + dt.timedelta(days=1)
    return chunks


def build_url(
    station: str,
    start: dt.date,
    end: dt.date,
    *,
    resolution: str,
    force: bool = True,
) -> str:
    """Build the official NEST ASCII query URL."""
    params: list[tuple[str, str]] = [
        ("wget", "1"),
        ("stations[]", station),
        ("output", "ascii"),
        ("tabchoice", nmdb_table_choice(station)),
        ("dtype", "corr_for_efficiency"),
        ("date_choice", "bydate"),
        ("start_year", str(start.year)),
        ("start_month", str(start.month)),
        ("start_day", str(start.day)),
        ("start_hour", "00"),
        ("start_min", "00"),
        ("end_year", str(end.year)),
        ("end_month", str(end.month)),
        ("end_day", str(end.day)),
        ("end_hour", "23"),
        ("end_min", "59"),
        ("tresolution", str(RESOLUTIONS[resolution])),
        ("yunits", "0"),
        ("smoothval", "0"),
        ("display_null", "1"),
    ]
    if force:
        params.append(("force", "1"))
    return f"{NMDB_URL}?{urllib.parse.urlencode(params)}"


def parse_summary(text: str) -> dict[str, str]:
    summary: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^#\s*([^:]+):\s*(.*?)\s*$", line)
        if match:
            summary[match.group(1).strip().upper()] = match.group(2).strip()
    return summary


def data_lines(text: str) -> list[str]:
    """Return non-comment data lines, excluding the column header."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [
        line
        for line in lines
        if not line.startswith("#")
        and not line.lower().startswith("start_date_time")
    ]


def parse_duration_minutes(value: str) -> int | None:
    for match in re.finditer(
        r"(\d+(?:\.\d+)?)\s*(min|hour|day)", value.lower()
    ):
        amount = float(match.group(1))
        factor = {"min": 1, "hour": 60, "day": 1440}[match.group(2)]
        minutes = amount * factor
        if minutes > 0 and minutes.is_integer():
            return int(minutes)
    return None


def validate_response(
    text: str, station: str, resolution: str
) -> tuple[dict[str, str], int]:
    if "sorry, no data available" in text.lower():
        raise NMDBNoData(f"NMDB reports no data for {station}")
    if "QUERY RESULTS SUMMARY" not in text:
        excerpt = " ".join(text.split())[:240]
        raise NMDBError(f"response has no NEST summary: {excerpt!r}")

    summary = parse_summary(text)
    if summary.get("STATION") != station:
        raise NMDBError(
            f"expected station {station}, got {summary.get('STATION', '<missing>')}"
        )
    if "corr_for_efficiency" not in summary.get("DATA TYPE", "").lower():
        raise NMDBError(f"unexpected data type: {summary.get('DATA TYPE')!r}")
    expected_table = nmdb_table_name(station)
    if summary.get("NMDB TABLE", "").lower() != expected_table.lower():
        raise NMDBError(
            f"unexpected NMDB table: {summary.get('NMDB TABLE')!r}; "
            f"expected {expected_table!r}"
        )

    averaging = summary.get("AVERAGING", "").strip()
    original_resolution = summary.get("ORIGINAL RES", "").strip()
    if parse_duration_minutes(original_resolution) is None:
        raise NMDBError(
            f"missing or invalid original resolution: {original_resolution!r}"
        )

    if resolution == "best":
        if averaging.lower() != "no":
            raise NMDBError(
                "NEST averaged a best-resolution request; "
                f"reported averaging is {averaging!r}"
            )
    else:
        requested_minutes = RESOLUTIONS[resolution]
        if averaging.lower() == "no":
            returned_minutes = parse_duration_minutes(original_resolution)
        elif "/" in averaging:
            returned_minutes = parse_duration_minutes(averaging.split("/", 1)[1])
        else:
            returned_minutes = None
        if returned_minutes != requested_minutes:
            raise NMDBError(
                f"NEST did not return {requested_minutes}-minute data; "
                f"reported averaging is {averaging!r} and original resolution "
                f"is {original_resolution!r}"
            )

    rows = data_lines(text)
    if not rows:
        raise NMDBError("NEST returned no data rows")
    for row in rows[: min(10, len(rows))]:
        fields = row.split(";")
        if len(fields) < 2:
            raise NMDBError(f"unexpected data row: {row!r}")
        try:
            dt.datetime.fromisoformat(fields[0].strip())
        except ValueError as exc:
            raise NMDBError(f"invalid timestamp in row: {row!r}") from exc
    return summary, len(rows)


def fetch(url: str, timeout: int) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "FluxPredict-NMDB-downloader/1.0",
            "Accept": "text/plain,text/html;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=build_ssl_context()
        ) as response:
            raw = response.read()
    except Exception as exc:  # urllib exposes several platform-specific errors
        raise NMDBRequestError(f"request failed: {exc}") from exc
    return raw.decode("utf-8", errors="replace")


def build_ssl_context() -> ssl.SSLContext:
    """Build a verified TLS context, including common macOS CA locations."""
    candidates: list[Path] = []
    default_cafile = ssl.get_default_verify_paths().cafile
    if default_cafile:
        candidates.append(Path(default_cafile))
    candidates.extend(
        [
            Path("/etc/ssl/cert.pem"),
            Path("/etc/ssl/certs/ca-certificates.crt"),
        ]
    )
    try:
        import certifi  # type: ignore

        candidates.append(Path(certifi.where()))
    except ImportError:
        try:
            from pip._vendor import certifi  # type: ignore

            candidates.append(Path(certifi.where()))
        except (ImportError, AttributeError):
            pass

    for cafile in candidates:
        if cafile.is_file():
            return ssl.create_default_context(cafile=str(cafile))
    return ssl.create_default_context()


def output_name(
    station: str, start: dt.date, end: dt.date, resolution: str
) -> str:
    suffix = "best" if resolution == "best" else f"{resolution}min"
    return f"{station}_{start:%Y%m%d}_{end:%Y%m%d}_{suffix}.txt"


def no_data_name(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".no_data")


def no_data_marker_matches_table(path: Path, station: str) -> bool:
    """Return whether a marker was produced for the station's current table."""
    text = path.read_text(encoding="utf-8", errors="replace")
    marker_table = parse_summary(text).get("NMDB TABLE", "").lower()
    expected_table = nmdb_table_name(station).lower()
    if marker_table:
        return marker_table == expected_table
    # Markers created before table metadata was added used revised original.
    return expected_table == "revised original"


def write_atomic(path: Path, text: str) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def parse_stations(value: str) -> list[str]:
    stations = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not stations:
        raise argparse.ArgumentTypeError("station list must not be empty")
    invalid = [
        item for item in stations if not re.fullmatch(r"[A-Z0-9]{4,5}", item)
    ]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"invalid station code(s): {', '.join(invalid)}"
        )
    return stations


def select_stations(args: argparse.Namespace) -> list[str]:
    if args.stations is not None:
        return args.stations
    if args.station_index is not None:
        if not 1 <= args.station_index <= len(DEFAULT_STATIONS):
            raise SystemExit(
                f"--station-index must be between 1 and {len(DEFAULT_STATIONS)}"
            )
        return [DEFAULT_STATIONS[args.station_index - 1]]
    if args.group_index is not None:
        if args.group_size is None:
            raise SystemExit("--group-index requires --group-size")
        if args.group_size < 1:
            raise SystemExit("--group-size must be at least 1")
        group_count = (len(DEFAULT_STATIONS) + args.group_size - 1) // args.group_size
        if not 1 <= args.group_index <= group_count:
            raise SystemExit(
                f"--group-index must be between 1 and {group_count} "
                f"when --group-size is {args.group_size}"
            )
        first = (args.group_index - 1) * args.group_size
        return list(DEFAULT_STATIONS[first : first + args.group_size])
    if args.group_size is not None:
        raise SystemExit("--group-size requires --group-index")
    return list(DEFAULT_STATIONS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download corrected NMDB data in validated resolution chunks."
    )
    station_selection = parser.add_mutually_exclusive_group()
    station_selection.add_argument(
        "--stations",
        type=parse_stations,
        help="comma-separated station codes",
    )
    station_selection.add_argument(
        "--station-index",
        type=int,
        help=f"one station by 1-based index (1-{len(DEFAULT_STATIONS)})",
    )
    station_selection.add_argument(
        "--group-index",
        type=int,
        help="1-based group index; requires --group-size",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        help="number of consecutive stations in an indexed group",
    )
    parser.add_argument(
        "--list-stations",
        action="store_true",
        help="print the 1-based station index and exit",
    )
    parser.add_argument("--start", type=parse_date, default=DEFAULT_START)
    end_selection = parser.add_mutually_exclusive_group()
    end_selection.add_argument("--end", type=parse_date)
    end_selection.add_argument(
        "--latest",
        action="store_true",
        help="download through the latest complete --chunk-months block",
    )
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=3,
        help="months per request (default: 3)",
    )
    parser.add_argument(
        "--resolution",
        choices=tuple(RESOLUTIONS),
        default="best",
        help="requested minutes, or 'best' for finest native resolution (default)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: rawdata/nmdb_best or rawdata/nmdb_<minutes>min",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="retries after a transport failure (default: 3)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="initial retry delay in seconds; doubled after each failure",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="seconds between sequential requests (default: 1)",
    )
    parser.add_argument(
        "--limit", type=int, help="download at most this many tasks; useful for testing"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--download",
        action="store_true",
        help="perform requests; without this flag only print the task plan",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.list_stations:
        for index, station in enumerate(DEFAULT_STATIONS, start=1):
            print(f"{index:2d}  {station}")
        return 0
    if args.chunk_months < 1:
        raise SystemExit("--chunk-months must be at least 1")
    if args.latest:
        try:
            args.end = latest_complete_chunk_end(
                args.start, args.chunk_months
            )
        except ValueError as exc:
            raise SystemExit(f"--latest: {exc}") from exc
        if args.end < args.start:
            raise SystemExit(
                "no complete chunk is available yet for the selected "
                "--start and --chunk-months"
            )
    elif args.end is None:
        args.end = DEFAULT_END
    if args.start > args.end:
        raise SystemExit("--start must not be after --end")
    if args.sleep < 0:
        raise SystemExit("--sleep must not be negative")
    if args.retries < 0:
        raise SystemExit("--retries must not be negative")
    if args.retry_delay < 0:
        raise SystemExit("--retry-delay must not be negative")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.output_dir is None:
        dirname = (
            "nmdb_best"
            if args.resolution == "best"
            else f"nmdb_{args.resolution}min"
        )
        args.output_dir = Path(__file__).resolve().parent / "rawdata" / dirname

    stations = select_stations(args)
    chunks = iter_chunks(args.start, args.end, args.chunk_months)
    tasks = [
        (station, chunk_start, chunk_end)
        for station in stations
        for chunk_start, chunk_end in chunks
    ]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    print(f"NMDB endpoint: {NMDB_URL}")
    print(f"Stations: {', '.join(stations)}")
    print(f"Date range: {args.start} to {args.end} (UTC)")
    print(f"Resolution: {args.resolution}")
    table_summary = ", ".join(
        f"{station}={nmdb_table_name(station)}" for station in stations
    )
    print(f"NMDB tables: {table_summary}")
    print(f"Chunks: {len(chunks)} per station; tasks: {len(tasks)}")
    print(f"Output: {args.output_dir}")

    if not args.download:
        for index, (station, start, end) in enumerate(tasks, start=1):
            path = args.output_dir / output_name(
                station, start, end, args.resolution
            )
            print(f"[{index}/{len(tasks)}] {station} {start} to {end} -> {path}")
        print("Dry run only. Add --download to retrieve data.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    downloaded = 0
    no_data = 0
    skipped = 0
    for index, (station, start, end) in enumerate(tasks, start=1):
        path = args.output_dir / output_name(station, start, end, args.resolution)
        marker = no_data_name(path)
        if path.exists() and not args.overwrite:
            try:
                summary, rows = validate_response(
                    path.read_text(encoding="utf-8"), station, args.resolution
                )
            except (OSError, UnicodeError, NMDBError) as exc:
                print(f"[{index}/{len(tasks)}] invalid existing file {path}: {exc}")
                print("Use --overwrite to replace it.", file=sys.stderr)
                failures += 1
                continue
            print(
                f"[{index}/{len(tasks)}] skip {path.name} "
                f"({rows} rows, original={summary.get('ORIGINAL RES')}, "
                f"averaging={summary.get('AVERAGING')})"
            )
            skipped += 1
            continue
        if marker.exists() and not args.overwrite:
            try:
                marker_matches = no_data_marker_matches_table(marker, station)
            except OSError as exc:
                print(f"  FAILED to read no-data marker: {exc}", file=sys.stderr)
                failures += 1
                continue
            if marker_matches:
                print(
                    f"[{index}/{len(tasks)}] skip {station} {start} to {end} "
                    "(no data marker)"
                )
                no_data += 1
                continue
            print(
                f"[{index}/{len(tasks)}] retry {station} {start} to {end}: "
                "no-data marker belongs to a different NMDB table"
            )

        url = build_url(station, start, end, resolution=args.resolution)
        print(
            f"[{index}/{len(tasks)}] download {station} {start} to {end} "
            f"[{nmdb_table_name(station)}]"
        )
        try:
            for attempt in range(args.retries + 1):
                try:
                    text = fetch(url, args.timeout)
                    break
                except NMDBRequestError as exc:
                    if attempt >= args.retries:
                        raise
                    delay = args.retry_delay * (2**attempt)
                    print(
                        f"  transport failure: {exc}; retry "
                        f"{attempt + 1}/{args.retries} in {delay:g}s"
                    )
                    time.sleep(delay)
            summary, rows = validate_response(text, station, args.resolution)
            if marker.exists():
                marker.unlink()
            write_atomic(path, text)
            print(
                f"  saved {path.name}: {rows} rows; "
                f"original={summary.get('ORIGINAL RES')}; "
                f"averaging={summary.get('AVERAGING')}"
            )
            downloaded += 1
        except NMDBNoData as exc:
            marker_text = (
                "# NMDB no-data marker; this chunk is intentionally left missing.\n"
                f"# STATION: {station}\n"
                f"# NMDB TABLE: {nmdb_table_name(station)}\n"
                f"# START DATE: {start.isoformat()} UTC\n"
                f"# END DATE: {end.isoformat()} UTC\n"
                f"# MESSAGE: {exc}\n"
            )
            try:
                write_atomic(marker, marker_text)
            except OSError as marker_exc:
                print(
                    f"  FAILED to write no-data marker: {marker_exc}",
                    file=sys.stderr,
                )
                failures += 1
                continue
            print(f"  no data; recorded {marker.name}")
            no_data += 1
        except (NMDBError, OSError, UnicodeError) as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            failures += 1
        if index < len(tasks):
            time.sleep(args.sleep)

    print(
        f"Finished: downloaded={downloaded}, no_data={no_data}, "
        f"skipped={skipped}, failed={failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
