#!/usr/bin/env python3
"""Map which NMDB table covers which period, for every station.

Why this needs its own probe
----------------------------
NEST only hands back the table you asked for under narrow conditions, so the
obvious probe (one long range at daily resolution) is useless for measuring
coverage: it silently returns the 1-hour table whatever you requested. Measured
on station JUNG over 2015-01 (see the module docstring history below), the
returned table depends on ``tresolution`` like this::

    best -> revised original   (native 1 min, 44632 rows)
    60   -> revised original   (aggregated to 1 h, 744 rows)
    120  -> 1 hour validated   <-- substitution starts here
    180  -> 1 hour validated
    360  -> 1 hour validated
    720  -> 1 hour validated
    1440 -> 1 hour validated

At ``tresolution=60`` the requested table is returned *as-is even when the range
is only partially covered*: a ``revori`` request for 2001-2026 comes back
labelled "revised original" holding only the rows NMDB actually has (for JUNG
that is 2008-08-12 onward). One request per (station, table) is therefore enough
to read off the true coverage -- start, end, and every internal gap.

Note that "no data at all" is a distinct outcome: NEST answers such a request
with a "sorry, no data available" page rather than an empty table.

Outputs
-------
* raw NEST responses cached under ``data/nmdb_table_survey/`` (resumable)
* a per-station summary, per-quarter availability maps, and a gap listing
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import nmdb_download as D
from station_metadata import NMDB_TABLE_NAME

TABLES = ("revori", "1h", "ori")
PROBE_RESOLUTION = "60"  # the only non-native setting that preserves table identity


def quarter_bounds(year: int, quarter: int) -> tuple[dt.date, dt.date]:
    start_month = 3 * (quarter - 1) + 1
    start = dt.date(year, start_month, 1)
    if start_month == 10:
        end = dt.date(year, 12, 31)
    else:
        end = dt.date(year, start_month + 3, 1) - dt.timedelta(days=1)
    return start, end


def parse_days(text: str) -> tuple[dict[str, str], set[dt.date]]:
    """Extract the summary block and the set of days carrying data."""
    summary: dict[str, str] = {}
    days: set[dt.date] = set()
    for line in text.splitlines():
        if line.startswith("#"):
            if ":" in line and "|" not in line:
                key, _, value = line.lstrip("# ").partition(":")
                summary[key.strip()] = value.strip()
            continue
        if not line or ";" not in line:
            continue
        stamp = line.split(";", 1)[0].strip()
        try:
            days.add(dt.date.fromisoformat(stamp[:10]))
        except ValueError:
            continue
    return summary, days


def fetch_cached(
    station: str, table: str, start: dt.date, end: dt.date,
    cache_dir: Path, timeout: int, sleep: float, retries: int = 3,
) -> str:
    cache = cache_dir / f"{station}_{table}_{start}_{end}.txt"
    if cache.exists() and cache.stat().st_size > 0:
        return cache.read_text(errors="replace")
    url = D.build_url(
        station, start, end, resolution=PROBE_RESOLUTION,
        table=table, display_null="0",
    )
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            text = D.fetch(url, timeout)
            break
        except Exception as exc:  # noqa: BLE001 - reported after the last attempt
            last = exc
            if attempt < retries:
                time.sleep(2.0 * (2**attempt))
    else:
        raise D.NMDBError(f"{station} {table}: {type(last).__name__}: {last}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(text)
    time.sleep(sleep)
    return text


def describe(days: set[dt.date]) -> dict[str, object]:
    if not days:
        return {"n": 0, "first": None, "last": None, "holes": [], "hole_days": 0}
    ordered = sorted(days)
    holes: list[tuple[dt.date, dt.date]] = []
    hole_days = 0
    for prev, cur in zip(ordered, ordered[1:]):
        gap = (cur - prev).days - 1
        if gap > 0:
            holes.append((prev + dt.timedelta(days=1), cur - dt.timedelta(days=1)))
            hole_days += gap
    return {
        "n": len(days), "first": ordered[0], "last": ordered[-1],
        "holes": holes, "hole_days": hole_days,
    }


def quarter_map(days: set[dt.date], start: dt.date, end: dt.date) -> str:
    """One character per calendar quarter: # >=99%, + >=51%, - >=1%, . none."""
    out = []
    year, quarter = start.year, (start.month - 1) // 3 + 1
    while True:
        qs, qe = quarter_bounds(year, quarter)
        if qs > end:
            break
        lo, hi = max(qs, start), min(qe, end)
        total = (hi - lo).days + 1
        have = sum(1 for d in days if lo <= d <= hi)
        if have == 0:
            out.append(".")
        elif have >= 0.99 * total:
            out.append("#")
        elif have >= 0.51 * total:
            out.append("+")
        else:
            out.append("-")
        if quarter == 4:
            year, quarter = year + 1, 1
        else:
            quarter += 1
    return "".join(out)


def year_ruler(start: dt.date, end: dt.date) -> str:
    """A 4-char-per-year ruler so the 104-char maps stay readable."""
    marks = []
    for year in range(start.year, end.year + 1):
        marks.append(f"{year % 100:02d}" + "  ")
    return "".join(marks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", default=",".join(D.DEFAULT_STATIONS))
    parser.add_argument("--tables", default=",".join(TABLES))
    parser.add_argument("--start", type=D.parse_date, default=dt.date(2001, 1, 1))
    parser.add_argument("--end", type=D.parse_date, default=dt.date(2026, 6, 30))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/nmdb_table_survey"))
    parser.add_argument("--report", type=Path, default=Path("logs/table_coverage.md"))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    stations = [s.strip().upper() for s in args.stations.split(",") if s.strip()]
    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line, flush=True)
        lines.append(line)

    emit(f"# NMDB 表覆盖调研：{args.start} ~ {args.end}")
    emit()
    emit(f"探测方式：`tresolution={PROBE_RESOLUTION}`、`display_null=0`、"
         f"`dtype=corr_for_efficiency`，一站一表一次请求。")
    emit()

    results: dict[str, dict[str, dict[str, object]]] = {}
    day_sets: dict[tuple[str, str], set[dt.date]] = {}
    mismatches: list[str] = []
    for table in tables:
        emit(f"## 正在探测 {table} …")
        for station in stations:
            try:
                text = fetch_cached(
                    station, table, args.start, args.end,
                    args.cache_dir, args.timeout, args.sleep,
                )
            except Exception as exc:  # noqa: BLE001
                emit(f"  {station:6} {table:7} 请求失败：{type(exc).__name__}: {exc}")
                results.setdefault(station, {})[table] = {"error": str(exc)}
                continue
            if "sorry, no data available" in text.lower():
                summary, days = {}, set()
            else:
                summary, days = parse_days(text)
            info = describe(days)
            info["summary"] = summary
            got = summary.get("NMDB TABLE", "")
            if got and got.lower() != NMDB_TABLE_NAME[table].lower():
                mismatches.append(f"{station} {table} -> {got}")
            results.setdefault(station, {})[table] = info
            day_sets[(station, table)] = days
            emit(f"  {station:6} {table:7} {info['n']:>7} 天  "
                 f"{info['first']} ~ {info['last']}")
        emit()

    # ---------------------------------------------------------------- summary
    emit("## 汇总")
    emit()
    emit("| 站 | revori 覆盖 | revori 天数 | 1h 覆盖 | 1h 天数 | ori 覆盖 | ori 天数 |")
    emit("|---|---|---|---|---|---|---|")
    for station in stations:
        cells = []
        for table in ("revori", "1h", "ori"):
            info = results.get(station, {}).get(table, {})
            if "error" in info:
                cells += ["请求失败", "—"]
            elif not info.get("n"):
                cells += ["**无**", "0"]
            else:
                cells += [f"{info['first']} ~ {info['last']}", str(info["n"])]
        emit(f"| {station} | " + " | ".join(cells) + " |")
    emit()

    # ------------------------------------------------------------ quarter maps
    emit("## 逐季度可用性")
    emit()
    emit("图例：`#` ≥99% 的日子有值，`+` 51–98%，`-` 1–50%，`.` 无数据；"
         "每格一个季度，从 2001Q1 起共 104 格。")
    emit()
    emit("```")
    emit("年:  " + year_ruler(args.start, args.end))
    for station in stations:
        emit(f"{station}:")
        for table in tables:
            days = day_sets.get((station, table))
            if days is None:
                continue
            emit(f"  {table:7} {quarter_map(days, args.start, args.end)}")
    emit("```")
    emit()

    # ------------------------------------------------------------- conclusion
    emit("## 分类")
    emit()
    full, none, partial = [], [], []
    for station in stations:
        info = results.get(station, {}).get("revori", {})
        if "error" in info:
            continue
        if not info.get("n"):
            none.append(station)
        elif info["first"] <= args.start and info["last"] >= args.end and not info["holes"]:
            full.append(station)
        else:
            partial.append(station)
    emit(f"* **revori 全程无洞**（{len(full)} 站）：{', '.join(full) or '无'}")
    emit(f"* **完全没有 revori，只能用 1h**（{len(none)} 站）：{', '.join(none) or '无'}")
    emit(f"* **revori 部分覆盖**（{len(partial)} 站）：{', '.join(partial) or '无'}")
    emit()

    for station in partial + none:
        info = results.get(station, {}).get("revori", {})
        if not info.get("n"):
            continue
        emit(f"### {station} 的 revori")
        emit()
        emit(f"起点 `{info['first']}`，终点 `{info['last']}`，"
             f"共 {info['n']} 天，区间内空 {info['hole_days']} 天，"
             f"空档 {len(info['holes'])} 处。")
        for lo, hi in info["holes"][:15]:
            emit(f"* {lo} ~ {hi}（{(hi - lo).days + 1} 天）")
        if len(info["holes"]) > 15:
            emit(f"* …另有 {len(info['holes']) - 15} 处")
        emit()

    if mismatches:
        emit("## 异常：返回的表与请求不符")
        emit()
        for item in mismatches:
            emit(f"* {item}")
        emit()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print(f"\n报告已写入 {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
