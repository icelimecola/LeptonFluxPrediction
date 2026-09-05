"""Metadata for the neutron-monitor stations used in the paper."""

from __future__ import annotations


# Vertical geomagnetic cutoff rigidities listed by NMDB NEST, in GV.
# https://www.nmdb.eu/nest/help.php#howto
CUTOFF_RIGIDITY_GV: dict[str, float] = {
    "AATB": 5.90,
    "APTY": 0.65,
    "FSMT": 0.30,
    "INVK": 0.30,
    "JUNG": 4.49,
    "JUNG1": 4.49,
    "LMKS": 3.84,
    "MXCO": 8.28,
    "NAIN": 0.30,
    "NEWK": 2.40,
    "OULU": 0.81,
    "PSNM": 16.80,
    "PWNK": 0.30,
    "SOPB": 0.10,
    "SOPO": 0.10,
    "TERA": 0.01,
    "THUL": 0.30,
    "YKTK": 1.65,
}


# PSNM has no data in ``revised original``. Its finest validated NMDB source
# is the 1-hour table; all other stations continue to use revised original.
DEFAULT_NMDB_TABLE_CHOICE = "revori"
NMDB_TABLE_CHOICE: dict[str, str] = {
    "PSNM": "1h",
}
NMDB_TABLE_NAME = {
    "revori": "revised original",
    "1h": "1 hour validated",
}


def nmdb_table_choice(station: str) -> str:
    return NMDB_TABLE_CHOICE.get(station.upper(), DEFAULT_NMDB_TABLE_CHOICE)


def nmdb_table_name(station: str) -> str:
    return NMDB_TABLE_NAME[nmdb_table_choice(station)]


def stations_by_cutoff_rigidity(stations: list[str]) -> list[str]:
    """Sort stations by vertical geomagnetic cutoff rigidity, ascending.

    Stations with equal cutoff rigidity are ordered alphabetically by
    station code, keeping the result deterministic.
    """
    return sorted(
        dict.fromkeys(stations),
        key=lambda station: (CUTOFF_RIGIDITY_GV[station], station),
    )
