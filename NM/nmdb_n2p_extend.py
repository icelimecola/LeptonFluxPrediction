#!/usr/bin/env python3
"""Extend the AMS daily proton flux over the whole NM period with a trained n2p.

Takes one trained n2p run (one SAITS station group, one data-seed), feeds the
group's complete SAITS-imputed NM matrix (2001-01-01 .. 2026-06-30) through the
network and writes a daily 30-bin proton-flux series for every NM day. This is
the input for the next network (proton -> positron).

Inputs (resolved from --group, each overridable)
------------------------------------------------
  data/nmdb_imputed_<G>/nm_daily_imputed_counts.npy (+ .csv, dates + names)
  data/nmdb_n2p_<G>/seed<N>/          --seed N, else best_seed.json, else the only seed*/;
      n2p_summary.json                stations, data_seed, split fractions ...
      Model/**/<tag>_best.pt          best-val weights
      Model/**/scalers.npz            train-split scalers (newer runs only)
      Error/**/<tag>_test_pred_flux.npy   used for the self-check
  PROTON/data/proton_allbin_dropna.npy (+ _meta.npz)   to rebuild the scalers
  PROTON/data/proton_allbin_meta.npz                    AMS measured, full calendar
  data/nmdb_filter_kde_oulu/data/nm_daily_oulu_filtered_counts.csv
                                      pre-SAITS matrix, to flag imputed inputs

Scalers
-------
Older n2p runs did not save their MinMax scalers. They are then rebuilt exactly
as in training: the same aligned (NM, proton) rows, the same two-step random
split (data_seed, test/val fractions), min/max fitted on the train rows only.
Either way a self-check re-predicts the test rows and must reproduce the saved
``*_test_pred_flux.npy``; a mismatch stops the script.

Outputs (data/nmdb_n2p_extend_<G>/)
-----------------------------------
  proton_extend_flux.npy    (T, n_bins) predicted flux, every NM day
  proton_extend_flux.csv    date + one column per rigidity bin
  proton_extend_meta.npz    same layout as PROTON/data/proton_allbin_meta.npz
                            (dates, rigidity_min, rigidity_max, flux) plus:
      flux_pred             = flux (the prediction)
      flux_ams              AMS measured flux, NaN where AMS has no value
      ams_n_bins            number of AMS bins measured that day (0..n_bins)
      period                1 = inside AMS range and measured,
                            2 = inside AMS range but an AMS gap (interpolation),
                            0 = outside the AMS range (extrapolation in time)
      imputed_mask          (T, N) True where that station's input was imputed
      n_imputed_inputs      imputed stations per day (-1 if unknown)
      input_out_of_range    any station outside the train min..max that day
      out_of_range_amount   largest excess, in train-scaled units (0 = inside)
      stations              the input stations, in model order
  extend_summary.json
  plots/per_bin/bin_XX_*.pdf           one bin per figure, with a residual
                                       panel pred/AMS-1 underneath; its band is
                                       AMS's day-to-day error sqrt(stat^2 +
                                       time-dependent syst^2), as AMS uses for
                                       its periodicity analysis
  plots/combined_extend_group_N.pdf    6 bins per 3x2 figure (no residual)
  plots/extend_panels_bins_1_7_13_19.pdf, extend_panels_bins_3_9_15_21.pdf
                                       4 stacked panels per figure, same bins
                                       as nmdb_n2p.py's test_panels (no residual)
      all drawn as lines: AMS measured as a thin red line inside a light-red
      total error band sqrt(stat^2 + total syst^2) (gaps left open), n2p prediction as a thin blue line
      on every day, on top; AMS range as dotted lines
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

import nmdb_n2p as n2p

BASE = Path(__file__).resolve().parent            # .../LeptonFluxPrediction/NM
PROTON_DIR = BASE.parent / "PROTON" / "data"
DEFAULT_GROUP = "lowrc9lj"
PLOT_GROUP_SIZE = 6
AMS_COLOR = "#d62728"          # same red as nmdb_n2p.py
AMS_BAND_COLOR = "#f4a6a6"     # light red: AMS stat+syst error band
PRED_ALPHA = 0.75              # default of --pred-alpha: top-layer prediction line opacity
PRED_COLOR = "#1a73e8"


# --------------------------------------------------------------------------
# locating the inputs
# --------------------------------------------------------------------------
def find_one(paths: list[Path], what: str, where: Path) -> Path:
    if len(paths) != 1:
        found = ", ".join(str(p) for p in paths) or "nothing"
        raise SystemExit(f"expected exactly one {what} under {where}, found: {found}")
    return paths[0]


def run_file(run_dir: Path, sub: str, pattern: str, required: bool = True):
    """One file of the run, current layout first.

    Current nmdb_n2p.py writes <run>/Model/, <run>/Error/; older runs used
    <run>/Model/n2p/, <run>/Error/n2p/. A re-run into an old directory can leave
    both, so the top level wins and the files of one layout are never mixed.
    """
    for folder in (run_dir / sub, run_dir / sub / "n2p"):
        hits = sorted(folder.glob(pattern))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise SystemExit(f"more than one {pattern} in {folder}: {hits}")
    if required:
        raise SystemExit(f"no {pattern} under {run_dir / sub}")
    return None


def resolve_run(n2p_dir: Path, seed: int | None = None) -> tuple[Path, dict]:
    """Pick the run: --seed, else best_seed.json, else the only seed*/."""
    if seed is None:
        best = n2p_dir / "best_seed.json"
        if best.is_file():
            seed = int(json.loads(best.read_text(encoding="utf-8"))["best_seed"])
            print(f"using best_seed.json -> seed{seed}")
    if seed is not None:
        summary_path = n2p_dir / f"seed{seed}" / "n2p_summary.json"
        if not summary_path.is_file():
            raise SystemExit(f"no {summary_path}")
    else:
        summaries = sorted(n2p_dir.glob("seed*/n2p_summary.json"))
        summary_path = find_one(summaries, "seed*/n2p_summary.json", n2p_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary_path.parent, summary


def resolve_local(path_text: str | None, fallback: Path) -> Path:
    """A path from the summary if it exists here, else the local default."""
    if path_text:
        candidate = Path(path_text)
        if not candidate.is_absolute():
            candidate = BASE / candidate
        if candidate.is_file():
            return candidate
    return fallback


# --------------------------------------------------------------------------
# scalers: saved by newer runs, rebuilt for older ones
# --------------------------------------------------------------------------
def safe_range(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    rng = hi - lo
    return np.where(rng == 0.0, 1.0, rng)          # sklearn's zero-range rule


def split_indices(n: int, test_frac: float, val_frac: float, data_seed: int):
    """Same two-step split as nmdb_n2p.py (sklearn train_test_split, shuffled)."""
    all_idx = np.arange(n)
    try:
        from sklearn.model_selection import train_test_split

        train_idx, test_idx = train_test_split(
            all_idx, test_size=test_frac, shuffle=True, random_state=data_seed)
        train_idx, val_idx = train_test_split(
            train_idx, test_size=val_frac / (1.0 - test_frac), shuffle=True,
            random_state=data_seed)
    except ImportError:
        # identical to ShuffleSplit for a float test_size (checked on the data)
        def once(idx, frac):
            n_test = math.ceil(frac * len(idx))
            perm = np.random.RandomState(data_seed).permutation(len(idx))
            return idx[perm[n_test:]], idx[perm[:n_test]]

        train_idx, test_idx = once(all_idx, test_frac)
        train_idx, val_idx = once(train_idx, val_frac / (1.0 - test_frac))
    return train_idx, val_idx, test_idx


def load_or_rebuild_scalers(run_dir, summary, neutron, neutron_dates, indices,
                            proton_path, proton_meta_path):
    ckpt = run_file(run_dir, "Model", "*_best.pt")
    scaler_file = ckpt.parent / "scalers.npz"          # same layout as the model
    saved = [scaler_file] if scaler_file.is_file() else []
    # the rows/split are needed for the self-check in both cases
    proton, proton_dates = n2p.read_proton(proton_path, proton_meta_path)
    rows = n2p.align_rows(neutron_dates, proton_dates)
    X = neutron[rows][:, indices]
    y_log = np.log10(proton)
    train_idx, _val_idx, test_idx = split_indices(
        len(X), summary["test_frac"], summary["val_frac"], summary["data_seed"])
    if saved:
        with np.load(saved[0]) as sc:
            scalers = {k: np.asarray(sc[k]) for k in
                       ("x_data_min", "x_data_max", "y_data_min", "y_data_max")}
        source = f"saved ({saved[0]})"
    else:
        scalers = {
            "x_data_min": X[train_idx].min(axis=0), "x_data_max": X[train_idx].max(axis=0),
            "y_data_min": y_log[train_idx].min(axis=0), "y_data_max": y_log[train_idx].max(axis=0),
        }
        source = "rebuilt from the training split"
    return scalers, source, X[test_idx], proton[test_idx]


def scale_x(X, sc):
    return (X - sc["x_data_min"]) / safe_range(sc["x_data_min"], sc["x_data_max"])


def unscale_y(Ys, sc):
    log10 = Ys * safe_range(sc["y_data_min"], sc["y_data_max"]) + sc["y_data_min"]
    return np.power(10.0, log10)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def load_model(run_dir, summary, n_bins, device):
    import torch

    ckpt = run_file(run_dir, "Model", "*_best.pt")
    model, kernels = n2p.build_model(len(summary["stations"]), n_bins, summary["dropout"])
    if summary.get("kernels") and list(summary["kernels"]) != list(kernels):
        raise SystemExit(f"kernel mismatch: summary {summary['kernels']} vs built {kernels}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    return model, ckpt


def predict(model, X_scaled, sc, device, batch=4096):
    import torch

    out = []
    with torch.no_grad():
        for start in range(0, len(X_scaled), batch):
            xb = torch.as_tensor(X_scaled[start:start + batch][..., None],
                                 dtype=torch.float32, device=device)
            out.append(model(xb).cpu().numpy())
    return unscale_y(np.concatenate(out, axis=0), sc)


# --------------------------------------------------------------------------
# flags
# --------------------------------------------------------------------------
def ams_on_grid(dates, full_meta_path, n_bins):
    flux_ams = np.full((len(dates), n_bins), np.nan)
    with np.load(full_meta_path) as meta:
        ams_dates = [dt.date.fromisoformat(str(d)) for d in meta["dates"]]
        ams_flux = np.asarray(meta["flux"], dtype=np.float64)   # NaN = no data
    if ams_flux.shape[1] != n_bins:
        raise SystemExit(f"AMS meta has {ams_flux.shape[1]} bins, model predicts {n_bins}")
    lookup = {d: i for i, d in enumerate(dates)}
    for k, day in enumerate(ams_dates):
        if day in lookup:
            flux_ams[lookup[day]] = ams_flux[k]
    ams_n_bins = np.isfinite(flux_ams).sum(axis=1).astype(np.int16)
    in_range = np.array([ams_dates[0] <= d <= ams_dates[-1] for d in dates])
    period = np.where(~in_range, 0, np.where(ams_n_bins > 0, 1, 2)).astype(np.int8)
    return flux_ams, ams_n_bins, period, (ams_dates[0], ams_dates[-1])


# AMS daily-proton error columns:
#   1 = flux_err_statistical, 2 = flux_err_timedependent (part of 3),
#   3 = flux_err_systematic_total (= 2 (+) the time-independent part)
#   "total": sqrt(1^2 + 3^2) -- the usual error of a measured point (flux plots)
#   "daily": sqrt(1^2 + 2^2) -- day-to-day comparison, as AMS uses for its
#            periodicity analysis; the time-independent part is a common offset
#            that the n2p model learns from AMS, so it cancels in pred/AMS - 1
AMS_ERR_COLUMNS = {
    "total": ("flux_err_statistical", "flux_err_systematic_total"),
    "daily": ("flux_err_statistical", "flux_err_timedependent"),
}


def ams_err_on_grid(dates, full_meta_path, n_bins, kind="total"):
    """AMS error on the NM grid, sqrt(a^2 + b^2) of the two columns of ``kind``
    (see AMS_ERR_COLUMNS); None if the meta file lacks them."""
    col_a, col_b = AMS_ERR_COLUMNS[kind]
    with np.load(full_meta_path) as meta:
        if not {col_a, col_b} <= set(meta.files):
            return None
        ams_dates = [dt.date.fromisoformat(str(d)) for d in meta["dates"]]
        a = np.asarray(meta[col_a], dtype=np.float64)
        b = np.asarray(meta[col_b], dtype=np.float64)
    err = np.full((len(dates), n_bins), np.nan)
    lookup = {d: i for i, d in enumerate(dates)}
    combined = np.sqrt(a ** 2 + b ** 2)
    for k, day in enumerate(ams_dates):
        if day in lookup:
            err[lookup[day]] = combined[k]
    return err


def imputed_inputs(dates, stations, original_csv):
    """(T, N) mask of inputs that were NaN before SAITS; None if unavailable."""
    if not original_csv.is_file():
        return None
    with original_csv.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = [h.strip() for h in next(reader)]
        rows = {dt.date.fromisoformat(r[0]): r for r in reader if r}
    missing_cols = [s for s in stations if s not in header]
    if missing_cols:
        raise SystemExit(f"{original_csv} lacks stations {missing_cols}")
    cols = [header.index(s) for s in stations]
    mask = np.zeros((len(dates), len(stations)), dtype=bool)
    for t, day in enumerate(dates):
        row = rows.get(day)
        if row is None:
            mask[t] = True
            continue
        mask[t] = [not row[c].strip() for c in cols]
    return mask


# --------------------------------------------------------------------------
# plots (same look as the SAITS daily series)
# --------------------------------------------------------------------------
def draw_ams_and_pred(axis, dates, pred, ams, ams_err, j, thin=False):
    """AMS as a thin solid red line inside a light-red stat+syst error band
    (both break at AMS's NaN gaps), the n2p prediction as a thin blue line on
    every day, drawn on top."""
    if ams_err is not None:
        axis.fill_between(dates, ams[:, j] - ams_err[:, j], ams[:, j] + ams_err[:, j],
                          color=AMS_BAND_COLOR, alpha=0.7, linewidth=0, zorder=1,
                          label="AMS stat+syst error")
    axis.plot(dates, ams[:, j], linestyle="-", linewidth=0.5 if thin else 0.6,
              color=AMS_COLOR, zorder=2, label="AMS Data")
    axis.plot(dates, pred[:, j], linestyle="-", linewidth=0.5 if thin else 0.6,
              color=PRED_COLOR, alpha=PRED_ALPHA, zorder=3, label="n2p prediction")


def style_time_axis(axis, dates, ams_range, mdates, labelsize):
    for edge in ams_range:
        axis.axvline(edge, color="0.3", linestyle=":", linewidth=1.0, zorder=0)
    axis.set_xlim(dates[0], dates[-1])
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.tick_params(direction="in", top=True, right=True, labelsize=labelsize)


def draw_panel(axis, dates, pred, ams, bins, j, ams_range, mdates, combined,
               ams_err=None):
    draw_ams_and_pred(axis, dates, pred, ams, ams_err, j, thin=combined)
    style_time_axis(axis, dates, ams_range, mdates, 8 if combined else 10)
    axis.set_yscale("log")
    axis.set_ylabel("flux (m$^{-2}$sr$^{-1}$s$^{-1}$GV$^{-1}$)",
                    fontsize=9 if combined else 10)
    axis.text(0.02, 0.94, f"bin {j + 1}: {bins[j][0]:g}-{bins[j][1]:g} GV",
              transform=axis.transAxes, ha="left", va="top",
              fontsize=11 if combined else 13, fontweight="bold")
    axis.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90 if combined else 0.88),
                borderaxespad=0.3, fontsize=9, frameon=False, handlelength=1.6)


def draw_residual(axis, dates, pred, ams, ams_err, j, ams_range, mdates):
    """pred/AMS - 1 on the AMS days (blue line) over AMS's day-to-day relative
    error +-sqrt(stat^2 + time-dependent syst^2)/AMS as a light-red band around 0
    (edges redrawn on top so the band stays visible where the residual covers
    it). ``ams_err`` here is the "daily" error. Used by the per-bin figures."""
    rel = pred[:, j] / ams[:, j] - 1.0                  # NaN where no AMS
    if ams_err is not None:
        rel_err = ams_err[:, j] / ams[:, j]
        axis.fill_between(dates, -rel_err, rel_err, color=AMS_BAND_COLOR,
                          alpha=0.7, linewidth=0, zorder=1,
                          label="AMS stat$\\oplus$time-dep. syst")
        axis.legend(loc="lower left", fontsize=8, frameon=False)
        for sign in (-1.0, 1.0):
            axis.plot(dates, sign * rel_err, linestyle="-", linewidth=0.6,
                      color=AMS_COLOR, alpha=0.8, zorder=4)
    axis.axhline(0.0, color="0.3", linewidth=0.8, zorder=2)
    axis.plot(dates, rel, linestyle="-", linewidth=0.5, color=PRED_COLOR,
              alpha=0.8, zorder=3)
    finite = np.isfinite(rel)
    if finite.any():
        axis.text(0.01, 0.92,
                  f"mean |pred/AMS-1| {np.mean(np.abs(rel[finite])) * 100:.2f}%",
                  transform=axis.transAxes, ha="left", va="top", fontsize=9)
        lim = max(0.02, float(np.nanpercentile(np.abs(rel), 99.5)) * 1.15)
        if ams_err is not None and np.isfinite(rel_err).any():
            # keep the whole error band in view, not only the residuals
            lim = max(lim, float(np.nanpercentile(rel_err, 99.5)) * 1.15)
        axis.set_ylim(-lim, lim)
    axis.set_ylabel("pred/AMS$-$1", fontsize=10)
    style_time_axis(axis, dates, ams_range, mdates, 10)


# same bin sets as nmdb_n2p.make_test_diagnostic_plots (0-based)
PANEL_SETS = ((0, 6, 12, 18), (2, 8, 14, 20))


def make_panel_plots(out_dir, dates, pred, ams, ams_err, bins, ams_range, group,
                     plt, mdates):
    """4 stacked panels per bin set, like nmdb_n2p.py's test_panels, as lines."""
    paths = []
    n_bins = pred.shape[1]
    for panel_bins in PANEL_SETS:
        chosen = [j for j in panel_bins if j < n_bins]
        if not chosen:
            continue
        figure, axes = plt.subplots(len(chosen), 1, figsize=(12.0, 2.8 * len(chosen)),
                                    sharex=True, squeeze=False)
        for axis, j in zip(axes[:, 0], chosen):
            draw_ams_and_pred(axis, dates, pred, ams, ams_err, j)
            style_time_axis(axis, dates, ams_range, mdates, 9)
            axis.text(0.99, 0.06, f"[{bins[j][0]:g}, {bins[j][1]:g}] GV",
                      transform=axis.transAxes, ha="right", fontsize=10,
                      fontweight="bold")
        axes[0, 0].legend(loc="upper left", frameon=False, fontsize=9)
        axes[-1, 0].set_xlabel("year")
        figure.supylabel("proton flux (m$^{-2}$sr$^{-1}$s$^{-1}$GV$^{-1}$)", fontsize=11)
        figure.suptitle(f"n2p extended proton flux ({group})", fontsize=13,
                        fontweight="bold")
        figure.tight_layout()
        tag = "_".join(str(j + 1) for j in chosen)
        path = out_dir / f"extend_panels_bins_{tag}.pdf"
        figure.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        paths.append(str(path))
    return paths


def make_plots(out_dir, dates, pred, ams, bins, ams_range, group, ams_err=None,
               ams_err_daily=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    paths = []
    per_bin = out_dir / "per_bin"
    per_bin.mkdir(parents=True, exist_ok=True)
    n_bins = pred.shape[1]
    for j in range(n_bins):
        # per bin: flux on top, residual pred/AMS - 1 underneath
        figure, (axis, ax_r) = plt.subplots(
            2, 1, figsize=(11.0, 5.0), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
        )
        draw_panel(axis, dates, pred, ams, bins, j, ams_range, mdates, combined=False,
                   ams_err=ams_err)
        draw_residual(ax_r, dates, pred, ams, ams_err_daily, j, ams_range, mdates)
        ax_r.set_xlabel("Year")
        # explicit margins: tight_layout does not handle the hspace-tuned
        # gridspec (it only warns); bbox_inches="tight" trims the rest
        figure.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.10)
        path = per_bin / f"bin_{j + 1:02d}_{bins[j][0]:g}-{bins[j][1]:g}GV.pdf"
        figure.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        paths.append(str(path))
    for start in range(0, n_bins, PLOT_GROUP_SIZE):
        chosen = list(range(start, min(start + PLOT_GROUP_SIZE, n_bins)))
        number = start // PLOT_GROUP_SIZE + 1
        figure, axes = plt.subplots(3, 2, figsize=(16.0, 11.0), squeeze=False)
        for k, j in enumerate(chosen):
            draw_panel(axes[k // 2, k % 2], dates, pred, ams, bins, j, ams_range,
                       mdates, combined=True, ams_err=ams_err)
        for k in range(len(chosen), PLOT_GROUP_SIZE):
            axes[k // 2, k % 2].set_visible(False)
        figure.suptitle(
            f"n2p extended proton flux ({group}), bins {chosen[0] + 1}-{chosen[-1] + 1}"
            "   red: AMS Data (band: stat+syst error)   blue: n2p prediction",
            fontsize=15, fontweight="bold")
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        path = out_dir / f"combined_extend_group_{number}.pdf"
        figure.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        paths.append(str(path))
    paths.extend(make_panel_plots(out_dir, dates, pred, ams, ams_err, bins,
                                  ams_range, group, plt, mdates))
    return paths


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="Extend the daily proton flux over the NM period with a trained n2p.")
    parser.add_argument("--group", default=DEFAULT_GROUP,
                        help=f"SAITS/n2p station group tag (default: {DEFAULT_GROUP})")
    parser.add_argument("--imputed-dir", type=Path, default=None,
                        help="default: data/nmdb_imputed_<group>")
    parser.add_argument("--n2p-dir", type=Path, default=None,
                        help="default: data/nmdb_n2p_<group>")
    parser.add_argument("--seed", type=int, default=None,
                        help="data_seed run to use (seed<N>/); default: best_seed "
                             "from <n2p-dir>/best_seed.json (nmdb_n2p.py --select-seed), "
                             "else the only seed*/ directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: data/nmdb_n2p_extend_<group>")
    parser.add_argument("--proton", type=Path, default=PROTON_DIR / "proton_allbin_dropna.npy",
                        help="training proton matrix, used to rebuild the scalers")
    parser.add_argument("--proton-meta", type=Path,
                        default=PROTON_DIR / "proton_allbin_dropna_meta.npz")
    parser.add_argument("--ams-full-meta", type=Path,
                        default=PROTON_DIR / "proton_allbin_meta.npz",
                        help="AMS measured flux on the full calendar (NaN = no data)")
    parser.add_argument("--nm-original", type=Path,
                        default=BASE / "data" / "nmdb_filter_kde_oulu" / "data"
                        / "nm_daily_oulu_filtered_counts.csv",
                        help="pre-SAITS NM matrix, to flag which inputs were imputed")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--selfcheck-rtol", type=float, default=1e-3)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--pred-alpha", type=float, default=PRED_ALPHA,
                        help="opacity of the blue prediction line drawn on top of the "
                             f"AMS line and band, 0 < a <= 1 (default: {PRED_ALPHA})")
    return parser


def main() -> int:
    global PRED_ALPHA
    args = build_parser().parse_args()
    if not 0.0 < args.pred_alpha <= 1.0:
        raise SystemExit("--pred-alpha must be in (0, 1]")
    PRED_ALPHA = args.pred_alpha          # read by draw_ams_and_pred at plot time
    import torch

    group = args.group
    imputed_dir = args.imputed_dir or BASE / "data" / f"nmdb_imputed_{group}"
    n2p_dir = args.n2p_dir or BASE / "data" / f"nmdb_n2p_{group}"
    out_dir = args.output_dir or BASE / "data" / f"nmdb_n2p_extend_{group}"
    device = torch.device(args.device)

    run_dir, summary = resolve_run(n2p_dir, args.seed)
    stations = list(summary["stations"])
    print(f"group {group}: run {run_dir}  stations={','.join(stations)}  "
          f"data_seed={summary['data_seed']}")

    # NM: whole imputed matrix, dated by its CSV, columns in model order
    neutron_path = imputed_dir / "nm_daily_imputed_counts.npy"
    neutron, dates = n2p.read_neutron(neutron_path, None)
    names = n2p.read_neutron_station_names(neutron_path, neutron.shape[1])
    indices = n2p.select_station_indices(names, stations)
    if [names[i] for i in indices] != stations:
        raise SystemExit(f"station order mismatch: matrix {[names[i] for i in indices]} "
                         f"vs model {stations}")
    if not np.isfinite(neutron[:, indices]).all():
        raise SystemExit("imputed NM matrix still has NaN in the model stations")

    proton_path = resolve_local(summary.get("proton"), args.proton)
    proton_meta = args.proton_meta
    bins = n2p.read_bins(proton_meta)
    n_bins = len(bins)

    scalers, scaler_source, X_test, flux_test = load_or_rebuild_scalers(
        run_dir, summary, neutron, dates, indices, proton_path, proton_meta)
    print(f"scalers: {scaler_source}")
    model, ckpt = load_model(run_dir, summary, n_bins, device)
    print(f"model: {ckpt}")

    # self-check against the test predictions saved by nmdb_n2p.py
    # the saved test prediction must come from the same layout as the model
    pred_dir = run_dir / "Error" / ("n2p" if ckpt.parent.name == "n2p" else "")
    saved_pred = sorted(pred_dir.glob("*_test_pred_flux.npy"))
    check = {"status": "skipped (no saved test prediction)"}
    if saved_pred:
        ref = np.load(saved_pred[0])
        mine = predict(model, scale_x(X_test, scalers), scalers, device)
        worst = float(np.max(np.abs(mine / ref - 1.0)))
        check = {"file": str(saved_pred[0]), "max_rel_diff": worst,
                 "rtol": args.selfcheck_rtol}
        if ref.shape != mine.shape or worst > args.selfcheck_rtol:
            raise SystemExit(f"self-check FAILED: max relative diff {worst:.3g} "
                             f"> {args.selfcheck_rtol} ({saved_pred[0]})")
        check["status"] = "passed"
        print(f"self-check passed: max relative diff {worst:.2e}")
    else:
        print("WARNING: no saved *_test_pred_flux.npy, self-check skipped")

    # predict every NM day
    X_all = neutron[:, indices]
    X_scaled = scale_x(X_all, scalers)
    flux_pred = predict(model, X_scaled, scalers, device)

    # flags
    flux_ams, ams_n_bins, period, ams_range = ams_on_grid(dates, args.ams_full_meta, n_bins)
    mask = imputed_inputs(dates, stations, args.nm_original)
    if mask is None:
        print(f"WARNING: {args.nm_original} not found, imputed-input flags set to -1")
        mask = np.zeros((len(dates), len(stations)), dtype=bool)
        n_imputed = np.full(len(dates), -1, dtype=np.int16)
    else:
        n_imputed = mask.sum(axis=1).astype(np.int16)
    excess = np.maximum(X_scaled - 1.0, -X_scaled).max(axis=1)
    out_amount = np.clip(excess, 0.0, None)
    out_of_range = out_amount > 0.0

    # outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    date_text = np.asarray([d.isoformat() for d in dates])
    np.save(out_dir / "proton_extend_flux.npy", flux_pred)
    with (out_dir / "proton_extend_flux.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", *(f"{lo:g}-{hi:g}GV" for lo, hi in bins)])
        for t, day in enumerate(date_text):
            writer.writerow([day, *(f"{v:.8g}" for v in flux_pred[t])])
    np.savez(
        out_dir / "proton_extend_meta.npz",
        dates=date_text,
        rigidity_min=np.asarray([b[0] for b in bins]),
        rigidity_max=np.asarray([b[1] for b in bins]),
        flux=flux_pred, flux_pred=flux_pred, flux_ams=flux_ams,
        ams_n_bins=ams_n_bins, period=period,
        imputed_mask=mask, n_imputed_inputs=n_imputed,
        input_out_of_range=out_of_range, out_of_range_amount=out_amount,
        stations=np.asarray(stations),
    )

    plots = []
    if not args.no_plots:
        ams_err = ams_err_on_grid(dates, args.ams_full_meta, n_bins, "total")
        ams_err_daily = ams_err_on_grid(dates, args.ams_full_meta, n_bins, "daily")
        plots = make_plots(out_dir / "plots", dates, flux_pred, flux_ams, bins,
                           ams_range, group, ams_err=ams_err,
                           ams_err_daily=ams_err_daily)

    counts = {
        "days": len(dates),
        "ams_measured": int((period == 1).sum()),
        "ams_gap_interpolated": int((period == 2).sum()),
        "outside_ams_range": int((period == 0).sum()),
        "days_with_imputed_input": int((n_imputed > 0).sum()),
        "days_input_out_of_range": int(out_of_range.sum()),
        "days_input_out_of_range_outside_ams": int((out_of_range & (period == 0)).sum()),
        "max_out_of_range_amount": float(out_amount.max()),
    }
    measured = np.isfinite(flux_ams)
    agree = (float(np.mean(np.abs(flux_pred[measured] / flux_ams[measured] - 1.0)))
             if measured.any() else None)
    summary_out = {
        "group": group,
        "date_range": [date_text[0], date_text[-1]],
        "ams_range": [ams_range[0].isoformat(), ams_range[1].isoformat()],
        "stations": stations,
        "n_bins": n_bins,
        "n2p_run": str(run_dir),
        "checkpoint": str(ckpt),
        "data_seed": summary["data_seed"],
        "neutron": str(neutron_path),
        "scalers": scaler_source,
        "self_check": check,
        "pred_alpha": args.pred_alpha,
        "counts": counts,
        "mean_abs_rel_diff_vs_ams_all_measured_days": agree,
        "period_codes": {"1": "AMS measured", "2": "AMS gap (interpolated)",
                         "0": "outside AMS range (extrapolated)"},
        "outputs": {"flux_npy": str(out_dir / "proton_extend_flux.npy"),
                    "flux_csv": str(out_dir / "proton_extend_flux.csv"),
                    "meta": str(out_dir / "proton_extend_meta.npz"),
                    "plots": plots},
        "note": "flux = prediction on every day; AMS measurements are kept "
                "separately in flux_ams and are not merged in",
    }
    (out_dir / "extend_summary.json").write_text(
        json.dumps(summary_out, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print(f"outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
