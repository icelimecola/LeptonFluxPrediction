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
