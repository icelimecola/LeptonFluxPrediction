#!/usr/bin/env python3
"""Train a proton -> positron residual CNN (same network family as nmdb_n2p.py).

Input  X : daily proton flux, 30 rigidity bins, 2001-2026, from
           nmdb_n2p_extend.py (data/nmdb_n2p_extend_<G>/proton_extend_meta.npz).
           Spliced per (day, bin): the AMS measurement where it exists
           (flux_ams), otherwise the n2p prediction (flux_pred).
Target y : AMS daily positron flux from data/lepton/sdiat/cguan/posflux.root (hflux_2d),
           processed like POS/step2/dataproc_fluxpos.py -- first and last
           energy bins dropped (27 bins, 1.0-41.9 GeV), all-zero head/tail
           days trimmed, remaining zeros = missing -- but NOT interpolated:
           only days with every kept bin measured are used.

Training mirrors nmdb_n2p.py: log10 on both sides, MinMax fitted on the train
rows, random 80/10/10 split by --data-seed, Adamax, MSE + Keras-style L2 on
the output kernel, Keras BatchNorm defaults, early stopping on val loss, best
weights restored. The convolution runs along the proton rigidity axis.

--loss-weight err (as nmdb_n2p.py --loss-weight ams): chi^2-like weight per
(day, bin), w = (R_k ln10 / eps_ik)^2 / c, eps = positron relative error from
posflux.root (error / flux), R_k = max - min of log10 flux on the train rows,
c = mean of the raw weight over the train split. Val/test losses are weighted
the same way (early stopping consistent). Default 'none' = plain MSE.

--avg-window 3day|br: train on time-averaged windows instead of days. Windows
are those of nmdb_aggregate.py (aligned to BR 2426 = 2011-05-15, consecutive,
non-overlapping). Label = exposure-weighted window mean of the positron flux
(hexps-pos.root), error combined the same way; input = equal-weight mean of the
proton input over the window days that have a positron value; only windows with
every bin filled are used; split random by window. After training the model is
also run on every daily proton input of the training range ->
Error/<tag>_daily_pred.npz (dates, pred, daily obs, err); daily metrics in the
summary include days of the training windows (reference only).

Zhao et al. 2026 five metrics (R2 / MAE / MPE / RMSE / chi2dof) per split and
for all days, via nmdb_n2p.paper_metrics_by_split; sigma = the posflux.root
error (only one error column, stored under chi2dof_total; chi2dof_daily empty).

Outputs under --output-dir (default data/nmdb_p2pos_<G>/seed<N>/):
  Model/<tag>_best.pt, Model/scalers.npz
  Error/<tag>_{train,val,test}_{true,pred}_flux.npy, _rel_error.npy,
        <tag>_test_meta.npz  (test dates, proton-input flags)
  Figure/...   same figure set as nmdb_n2p.py (positron / energy labels), incl.
               test_relerr_vs_rigidity.pdf (+ _wide.pdf for slides): AMS positron
               error band (single ROOT error column, mean over the test days),
               mean +- std of pred/obs - 1, lower relative-uncertainty panel
  Error/<tag>_test_meta.npz also keeps the test-day positron relative error and
               the energy bins, so nmdb_n2p_replot_relerr.py can redraw it
  p2pos_summary.json, paper_metrics.{json,csv}
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path

import numpy as np

import nmdb_n2p as n2p

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # LeptonFluxPrediction/
import datapaths as dp  # noqa: E402  原始数据路径统一在 datapaths.py

BASE = Path(__file__).resolve().parent            # .../LeptonFluxPrediction/NM
DEFAULT_GROUP = "lowrc9lj"
# legend of the AMS positron error in the relative-difference figure
POS_ERR_LABEL = "AMS positron error (mean)"
# default y range +-ylim of the relative-difference figure (also used by
# nmdb_n2p_replot_relerr.py for positron runs)
RELERR_YLIM = 1.0


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def read_proton_spliced(meta_path: Path):
    """Proton input: AMS where measured, n2p prediction elsewhere (per bin)."""
    with np.load(meta_path) as meta:
        dates = [dt.date.fromisoformat(str(d)) for d in meta["dates"]]
        pred = np.asarray(meta["flux_pred"], dtype=np.float64)
        ams = np.asarray(meta["flux_ams"], dtype=np.float64)
        bins = list(zip(meta["rigidity_min"].tolist(), meta["rigidity_max"].tolist()))
        out_of_range = (np.asarray(meta["input_out_of_range"], dtype=bool)
                        if "input_out_of_range" in meta.files
                        else np.zeros(len(dates), dtype=bool))
    measured = np.isfinite(ams)
    flux = np.where(measured, ams, pred)
    if not np.isfinite(flux).all() or (flux <= 0).any():
        raise SystemExit("spliced proton flux has non-finite or non-positive values")
    return dates, flux, measured, out_of_range, bins


def read_proton_period(meta_path: Path):
    """``period`` of proton_extend_meta.npz: 1 = AMS PRL measured day, 2 = gap
    inside the AMS PRL range (n2p interpolation), 0 = outside (extrapolation)."""
    with np.load(meta_path) as meta:
        return np.asarray(meta["period"]) if "period" in meta.files else None


def read_positron_exposure(path: Path, prefix: str, dates, bins):
    """Daily exposure time per energy bin on ``dates`` (0 = none), from
    hexps-pos.root <prefix><lo>to<hi>GeV, as nmdb_aggregate.py."""
    import uproot
    import nmdb_aggregate as agg
    hexp = uproot.open(path)
    expo = np.zeros((len(dates), len(bins)))
    for j, (lo, hi) in enumerate(bins):
        h = hexp[f"{prefix}{lo:g}to{hi:g}GeV"]
        lk = {d: i for i, d in enumerate(agg.day_starts(h.axes[0].edges()))}
        vals = np.asarray(h.values(), dtype=np.float64)
        expo[:, j] = [vals[lk[d]] if d in lk else 0.0 for d in dates]
    return expo


def build_avg_windows(length, e_dates, e_flux, e_err, expo, p_dates, p_flux,
                      p_measured, p_oor, allowed):
    """Window samples as nmdb_aggregate.aggregate (see --avg-window)."""
    import nmdb_aggregate as agg
    lookup = {d: i for i, d in enumerate(p_dates)}
    in_range = np.asarray([d in lookup and bool(allowed[lookup[d]]) for d in e_dates])
    has = np.isfinite(e_flux) & (e_flux > 0) & (expo > 0) & in_range[:, None]
    day_any = has.any(axis=1)
    days_with = [d for d, h in zip(e_dates, day_any) if h]
    if not days_with:
        raise SystemExit("no positron day inside the proton training range")
    flux_m = np.where(has, e_flux, np.nan)
    e_lookup = {d: i for i, d in enumerate(e_dates)}
    rec = {k: [] for k in ("start", "end", "x", "y", "err", "n_days", "meas", "oor")}
    used_days = []
    n_all = 0
    for w in agg.make_windows(days_with[0], days_with[-1], length):
        rows = agg.window_rows(w, e_lookup)
        rows = rows[day_any[rows]] if rows.size else rows
        if not rows.size:
            continue
        n_all += 1
        f, sig, _tot, _n = agg.exposure_mean(flux_m, e_err, expo, rows)
        if not (np.isfinite(f).all() and np.isfinite(sig).all()):
            continue                                   # every bin needs a value
        p_rows = np.asarray([lookup[e_dates[i]] for i in rows])
        rec["start"].append(w["start"])
        rec["end"].append(w["end"])
        rec["x"].append(p_flux[p_rows].mean(axis=0))   # equal weight, linear flux
        rec["y"].append(f)
        rec["err"].append(sig)
        rec["n_days"].append(len(rows))
        rec["meas"].append(p_measured[p_rows].mean())
        rec["oor"].append(bool(p_oor[p_rows].any()))
        used_days.extend(e_dates[i] for i in rows)
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["n_windows_with_data"] = n_all
    out["first_day"], out["last_day"] = min(used_days), max(used_days)
    return out


def read_positron_root(path: Path):
    """posflux.root -> (dates, flux, err, bins); same cuts as dataproc_fluxpos.py."""
    try:
        import uproot
    except ImportError as exc:
        raise SystemExit("reading posflux.root needs uproot in the run env") from exc
    hist = uproot.open(path)["hflux_2d;1"]
    flux = np.asarray(hist.values(), dtype=np.float64)       # (days, 29)
    err = np.asarray(hist.errors(), dtype=np.float64)
    x_low = hist.axis(0).member("fXmin")
    x_high = hist.axis(0).member("fXmax")
    n_x = hist.axis(0).member("fNbins")
    width = (x_high - x_low) / n_x                             # 86400 s
    edges = np.asarray(hist.axis(1).member("fXbins"), dtype=np.float64)
    keep = np.arange(1, flux.shape[1] - 1)                     # drop first + last
    flux, err = flux[:, keep], err[:, keep]
    edges = edges[keep[0]:keep[-1] + 2]
    first = dt.datetime.fromtimestamp(x_low + width / 2, tz=dt.timezone.utc).date()
    dates = [first + dt.timedelta(days=i) for i in range(flux.shape[0])]
    rows = np.where(flux.sum(axis=1) > 0)[0]                   # trim zero head/tail
    flux, err = flux[rows[0]:rows[-1] + 1], err[rows[0]:rows[-1] + 1]
    dates = dates[rows[0]:rows[-1] + 1]
    missing = flux == 0                                        # zeros = missing
    flux[missing] = np.nan
    err[missing] = np.nan
    bins = [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]
    return dates, flux, err, bins


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the proton -> positron residual CNN (PyTorch).")
    parser.add_argument("--group", default=DEFAULT_GROUP,
                        help=f"station group of the proton extension (default {DEFAULT_GROUP})")
    parser.add_argument("--proton-extend", type=Path, default=None,
                        help="proton_extend_meta.npz (default: "
                             "data/nmdb_n2p_extend_<group>/proton_extend_meta.npz)")
    parser.add_argument("--positron-root", type=Path,
                        default=dp.POSFLUX_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: data/nmdb_p2pos_<group>/seed<data-seed>")
    parser.add_argument("--data-seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42, help="global seed (init etc.)")
    parser.add_argument("--quick", action="store_true",
                        help="smoke run: epochs<=500, output into <output-dir>/quick")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--train-range", choices=("ams", "all"), default="ams",
        help="training days: 'ams' (default) = only days inside the AMS PRL published "
             "range of the proton input (period 1/2: PRL measured where present, n2p "
             "interpolation in the gaps); 'all' = every aligned positron day, incl. "
             "n2p extrapolation after the PRL range (old behaviour)")
    parser.add_argument(
        "--avg-window", choices=("none", "3day", "br"), default="none",
        help="'none' = daily samples (default, old behaviour); '3day' / 'br' = "
             "train on non-overlapping 3-day / 27-day windows of nmdb_aggregate.py: "
             "exposure-weighted positron label, equal-weight proton input; the "
             "trained model is then also run on every daily input")
    parser.add_argument("--exposure-root", type=Path,
                        default=dp.HEXPS_POS_ROOT,
                        help="positron exposure time (for --avg-window)")
    parser.add_argument("--exposure-prefix", default="h1t/sf1/h1exp_igrf_T_fov25_sf1_ene")
    parser.add_argument(
        "--loss-weight", choices=("none", "err"), default="none",
        help="'none' = plain MSE (default, old behaviour); 'err' = chi^2-like "
             "weight per (day, bin) from the posflux.root error, as nmdb_n2p.py "
             "--loss-weight ams: w = (R_k ln10 / eps_ik)^2 / c, eps = err/flux, "
             "c = mean of the raw weight over the train split")
    parser.add_argument(
        "--init", choices=("keras", "torch"), default="keras",
        help="weight init, as nmdb_n2p.py: 'keras' = glorot_uniform + zero bias "
             "(default); 'torch' = PyTorch default")
    parser.add_argument(
        "--drop-last", action="store_true",
        help="drop the last incomplete training batch (old behaviour); default "
             "keeps it, as Keras model.fit and nmdb_n2p.py do")
    parser.add_argument(
        "--relerr-ylim", type=float, default=RELERR_YLIM,
        help="y range +-ylim of test_relerr_vs_rigidity.pdf, one tick per ylim/10 "
             "(default 1.0: the daily positron error is ~20-40%%, so +-0.5 cut "
             "~5%% of the test points; nmdb_n2p.py uses 0.1 for protons)")
    parser.add_argument(
        "--select-seed", action="store_true",
        help="no training: compare all <output-dir>/seed*/ runs, write "
             "seed_summary.csv, best_seed.json, seed_selection.pdf into <output-dir> "
             "(default <output-dir>: data/nmdb_p2pos_<group>)")
    parser.add_argument(
        "--select-by", choices=("val_loss", "val_mre", "test_mre"), default="val_loss",
        help="criterion for --select-seed (default val_loss)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.select_seed:
        group_dir = args.output_dir or BASE / "data" / f"nmdb_p2pos_{args.group}"
        return n2p.select_seed(group_dir, None, args.select_by,
                               summary_name="p2pos_summary.json", label="p2pos")
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    from sklearn.model_selection import train_test_split

    torch.manual_seed(args.seed)
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    proton_meta = args.proton_extend or (
        BASE / "data" / f"nmdb_n2p_extend_{args.group}" / "proton_extend_meta.npz")
    out_dir = args.output_dir or (
        BASE / "data" / f"nmdb_p2pos_{args.group}" / f"seed{args.data_seed}")
    if args.quick:
        args.epochs = min(args.epochs, 500)
        out_dir = out_dir / "quick"

    p_dates, p_flux, p_measured, p_oor, p_bins = read_proton_spliced(proton_meta)
    e_dates, e_flux, e_err, e_bins = read_positron_root(args.positron_root)
    print(f"proton input : {p_dates[0]} .. {p_dates[-1]}  bins={len(p_bins)}")
    print(f"positron     : {e_dates[0]} .. {e_dates[-1]}  bins={len(e_bins)} "
          f"({e_bins[0][0]:g}-{e_bins[-1][1]:g} GeV)")

    # training days: every positron bin measured, and inside the proton range;
    # with --train-range ams also inside the AMS PRL range (no extrapolated input)
    lookup = {d: i for i, d in enumerate(p_dates)}
    complete = np.isfinite(e_flux).all(axis=1)
    allowed = np.ones(len(p_dates), dtype=bool)
    if args.train_range == "ams":
        period = read_proton_period(proton_meta)
        if period is None:
            raise SystemExit("--train-range ams needs 'period' in the proton extend meta")
        allowed = period > 0
        inside = [p_dates[i] for i in np.where(allowed)[0]]
        print(f"train range  : AMS PRL range {inside[0]} .. {inside[-1]} "
              f"({int((period == 1).sum())} measured + {int((period == 2).sum())} gap days)")
    win = None
    if args.avg_window == "none":
        pos_rows = [k for k, d in enumerate(e_dates)
                    if complete[k] and d in lookup and allowed[lookup[d]]]
        if not pos_rows:
            raise SystemExit("no complete positron day inside the proton range")
        pro_rows = np.asarray([lookup[e_dates[k]] for k in pos_rows])
        pos_rows = np.asarray(pos_rows)
        X_raw = p_flux[pro_rows]
        y_raw = e_flux[pos_rows]
        y_err = e_err[pos_rows]
        day = np.asarray([e_dates[k] for k in pos_rows])
        meas_frac = p_measured[pro_rows].mean(axis=1)              # 1 = all bins AMS
        oor = p_oor[pro_rows]
        first_day, last_day = day[0], day[-1]
    else:
        import nmdb_aggregate as agg
        length = agg.SCALES[args.avg_window]
        expo = read_positron_exposure(args.exposure_root, args.exposure_prefix,
                                      e_dates, e_bins)
        win = build_avg_windows(length, e_dates, e_flux, e_err, expo, p_dates, p_flux,
                                p_measured, p_oor, allowed)
        X_raw, y_raw, y_err = win["x"], win["y"], win["err"]
        day = win["start"]                       # a window is dated by its first day
        meas_frac, oor = win["meas"], win["oor"]
        first_day, last_day = win["first_day"], win["last_day"]
        print(f"avg window   : {args.avg_window} ({length} days), {len(day)} complete "
              f"windows of {win['n_windows_with_data']} with data; label days "
              f"{first_day} .. {last_day}; days per window median "
              f"{np.median(win['n_days']):g}")
    n_days, n_in = X_raw.shape
    n_out = y_raw.shape[1]
    print(f"aligned {'windows' if win is not None else 'days'}={n_days}  "
          f"proton bins={n_in}  positron bins={n_out}")
    print(f"  proton input fully measured on {int((meas_frac == 1).sum())} days, "
          f"fully predicted on {int((meas_frac == 0).sum())}, mixed on "
          f"{int(((meas_frac > 0) & (meas_frac < 1)).sum())}")

    X_log, y_log = np.log10(X_raw), np.log10(y_raw)
    all_idx = np.arange(n_days)
    train_idx, test_idx = train_test_split(
        all_idx, test_size=args.test_frac, shuffle=True, random_state=args.data_seed)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=args.val_frac / (1.0 - args.test_frac), shuffle=True,
        random_state=args.data_seed)

    from sklearn.preprocessing import MinMaxScaler
    scaler_x = MinMaxScaler().fit(X_log[train_idx])
    scaler_y = MinMaxScaler().fit(y_log[train_idx])
    sets = {}
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        sets[name] = (scaler_x.transform(X_log[idx])[..., None],
                      scaler_y.transform(y_log[idx]), idx)

    # optional chi^2-like loss weight (same formula as nmdb_n2p.py --loss-weight ams):
    # w~_ik = (R_k ln10 / eps_ik)^2, w = w~ / c, c = mean of w~ on the train rows,
    # so w * (yhat - y)^2 = ((ln phat - ln p) / eps)^2 / c on the MinMax(log10) target
    use_w = args.loss_weight == "err"
    loss_weight_c, n_eps_filled = None, 0
    if use_w:
        eps = y_err / y_raw
        bad = ~np.isfinite(eps) | (eps <= 0)
        n_eps_filled = int(bad.sum())
        if n_eps_filled:
            med = np.nanmedian(np.where(bad, np.nan, eps), axis=0)
            eps = np.where(bad, med[None, :], eps)
            print(f"positron error: {n_eps_filled} missing/non-positive values "
                  "filled with bin median")
        R = scaler_y.data_max_ - scaler_y.data_min_
        w_raw = (R[None, :] * np.log(10.0) / eps) ** 2
        loss_weight_c = float(np.mean(w_raw[train_idx]))
        w_all = w_raw / loss_weight_c
        print(f"loss weight: err (posflux.root), c = {loss_weight_c:.6g}; per-bin mean "
              f"w (train): bin1 {w_all[train_idx, 0].mean():.3g}, "
              f"bin{n_out} {w_all[train_idx, -1].mean():.3g}")
    else:
        w_all = np.ones_like(y_log)

    model, kernels = n2p.build_model(n_in, n_out, args.dropout, init=args.init)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"kernels: {kernels}  trainable parameters: {n_params:,}")
    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.999), eps=1e-7)
    mse = torch.nn.MSELoss()

    def criterion(pred, target, weight=None):   # Keras L2(l2) on the output kernel only
        data = mse(pred, target) if weight is None else torch.mean(weight * (pred - target) ** 2)
        return data + args.l2 * torch.sum(model.head.weight ** 2)

    def tensor(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    Xv, yv = tensor(sets["val"][0]), tensor(sets["val"][1])
    Xt, yt = tensor(sets["test"][0]), tensor(sets["test"][1])
    wv = tensor(w_all[val_idx]) if use_w else None
    wt = tensor(w_all[test_idx]) if use_w else None
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.as_tensor(sets["train"][0], dtype=torch.float32),
            torch.as_tensor(sets["train"][1], dtype=torch.float32),
            torch.as_tensor(w_all[train_idx], dtype=torch.float32)),
        batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last)

    model_dir, fig_dir, err_dir = out_dir / "Model", out_dir / "Figure", out_dir / "Error"
    for path in (model_dir, fig_dir, err_dir):
        path.mkdir(parents=True, exist_ok=True)
    tag = (f"seed{args.data_seed}_p{n_in}e{n_out}_{args.epochs}ep_lr{args.lr}"
           f"_l2{args.l2}_do{args.dropout}_b{args.batch_size}"
           + ("_werr" if use_w else "")
           + ("" if win is None else f"_avg{args.avg_window}"))
    ckpt = model_dir / f"{tag}_best.pt"
    np.savez(model_dir / "scalers.npz",
             x_data_min=scaler_x.data_min_, x_data_max=scaler_x.data_max_,
             y_data_min=scaler_y.data_min_, y_data_max=scaler_y.data_max_,
             x_is_log10=np.asarray(True), y_is_log10=np.asarray(True))

    train_losses, val_losses, test_losses = [], [], []
    best_val, best_epoch, patience_left = float("inf"), -1, args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch = []
        for xb, yb, wb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb, wb.to(device) if use_w else None)
            loss.backward()
            optimizer.step()
            batch.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(Xv), yv, wv).item())
            test_loss = float(criterion(model(Xt), yt, wt).item())
        train_losses.append(float(np.mean(batch)))
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        improved = val_loss < best_val - args.min_delta
        if improved:
            best_val, best_epoch, patience_left = val_loss, epoch, args.patience
            torch.save(model.state_dict(), ckpt)
        else:
            patience_left -= 1
        if epoch == 1 or epoch % 10 == 0 or improved or patience_left == 0:
            print(f"epoch {epoch:5d} train {train_losses[-1]:.6f} val {val_loss:.6f} "
                  f"test {test_loss:.6f}" + (" *" if improved else ""))
        if patience_left == 0:
            print(f"early stopping at epoch {epoch}; best val {best_val:.6f} @ {best_epoch}")
            break
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"loaded best checkpoint: {ckpt.name} (epoch {best_epoch})")

    metrics, true_by, pred_by, date_splits = {}, {}, {}, {}
    for name, (X_in, _ys, idx) in sets.items():
        true_flux = y_raw[idx]
        pred_flux = n2p.predict_flux(model, X_in, scaler_y, device)
        rel = pred_flux / true_flux - 1.0
        true_by[name], pred_by[name] = true_flux, pred_flux
        date_splits[name] = day[idx]
        np.save(err_dir / f"{tag}_{name}_true_flux.npy", true_flux)
        np.save(err_dir / f"{tag}_{name}_pred_flux.npy", pred_flux)
        np.save(err_dir / f"{tag}_{name}_rel_error.npy", rel)
        ss_res = float(np.sum((pred_flux - true_flux) ** 2))
        ss_tot = float(np.sum((true_flux - true_flux.mean()) ** 2))
        metrics[name] = {
            "mean_abs_relative_error": float(np.mean(np.abs(rel))),
            "mean_signed_relative_error": float(np.mean(rel)),
            "per_bin_mean_abs_relative_error": np.mean(np.abs(rel), axis=0).tolist(),
            "per_bin_mean_signed_relative_error": np.mean(rel, axis=0).tolist(),
            "mae_log10": float(np.mean(np.abs(np.log10(pred_flux) - np.log10(true_flux)))),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        }
        print(name, f"MRE {metrics[name]['mean_abs_relative_error'] * 100:.3f}%  "
              f"signed {metrics[name]['mean_signed_relative_error'] * 100:+.3f}%")
    # Zhao et al. 2026 five metrics; sigma = posflux.root error (one column ->
    # chi2dof_total; chi2dof_daily stays empty)
    paper = n2p.paper_metrics_by_split(
        true_by, pred_by, {"train": train_idx, "val": val_idx, "test": test_idx},
        y_err, None)
    n2p.write_paper_metrics(out_dir, paper)
    for name in n2p.PAPER_METRIC_SPLITS:
        print("paper", name, json.dumps(paper[name]))
    np.savez(err_dir / f"{tag}_test_meta.npz",
             dates=np.asarray([d.isoformat() for d in day[test_idx]]),
             proton_measured_fraction=meas_frac[test_idx],
             proton_input_out_of_range=oor[test_idx],
             # for nmdb_n2p_replot_relerr.py: redraw without retraining
             positron_rel_err=y_err[test_idx] / y_raw[test_idx],
             energy_min=np.asarray([b[0] for b in e_bins]),
             energy_max=np.asarray([b[1] for b in e_bins]),
             # --avg-window: the window inputs, so nmdb_aggregate.py's self-check
             # can re-predict the test windows (absent for daily runs)
             **({} if win is None else {
                 "window_end": np.asarray([d.isoformat() for d in win["end"][test_idx]]),
                 "window_n_days": win["n_days"][test_idx],
                 "input_mean": X_raw[test_idx]}))

    # --avg-window: run the window-trained model on every daily input of the range
    daily = None
    if win is not None:
        d_rows = np.asarray([i for i, d in enumerate(p_dates)
                             if allowed[i] and first_day <= d <= last_day])
        Xd = scaler_x.transform(np.log10(p_flux[d_rows]))[..., None]
        pred_d = n2p.predict_flux(model, Xd, scaler_y, device)
        e_lookup = {d: k for k, d in enumerate(e_dates)}
        obs_d = np.full_like(pred_d, np.nan)
        err_d = np.full_like(pred_d, np.nan)
        for r, i in enumerate(d_rows):
            k = e_lookup.get(p_dates[i])
            if k is not None:
                obs_d[r], err_d[r] = e_flux[k], e_err[k]
        np.savez(err_dir / f"{tag}_daily_pred.npz",
                 dates=np.asarray([p_dates[i].isoformat() for i in d_rows]),
                 pred=pred_d, obs=obs_d, obs_err=err_d,
                 energy_min=np.asarray([b[0] for b in e_bins]),
                 energy_max=np.asarray([b[1] for b in e_bins]))
        full = np.isfinite(obs_d).all(axis=1)
        rel_d = pred_d[full] / obs_d[full] - 1.0
        daily = {
            "note": "window-trained model on daily inputs vs daily posflux.root; "
                    "includes days of the training windows -> reference only",
            "n_days_predicted": int(len(d_rows)), "n_days_complete_obs": int(full.sum()),
            "mean_abs_relative_error": float(np.mean(np.abs(rel_d))),
            "mean_signed_relative_error": float(np.mean(rel_d)),
            "per_bin_mean_abs_relative_error": np.mean(np.abs(rel_d), axis=0).tolist(),
            "per_bin_sum_pred_over_sum_obs_minus_1":
                (pred_d[full].sum(axis=0) / obs_d[full].sum(axis=0) - 1.0).tolist(),
        }
        print(f"daily (window model, all {int(full.sum())} complete days): MRE "
              f"{daily['mean_abs_relative_error'] * 100:.2f}%  signed "
              f"{daily['mean_signed_relative_error'] * 100:+.2f}%")

    plot_paths, ts_paths = [], []
    if not args.no_plots:
        n2p.RELERR_YLIM = args.relerr_ylim
        n2p.PLOT_LABELS.update(particle="positron", unit="GeV", axis="energy",
                               err="ROOT hist error")
        try:
            plot_paths = n2p.make_plots(fig_dir, train_losses, val_losses, test_losses,
                                        true_by["test"], pred_by["test"], e_bins)
        except Exception as exc:
            print(f"[skip plots] {exc}")
        try:
            ts_paths = n2p.make_timeseries_plots(fig_dir / "timeseries", fig_dir,
                                                 date_splits, true_by, pred_by, e_bins)
        except Exception as exc:
            print(f"[skip timeseries plots] {exc}")
        try:
            # relative-difference figure: AMS positron error band (the single
            # error column of posflux.root, mostly statistical) of the TEST
            # days, std of pred/obs - 1, lower relative-uncertainty panel and a
            # wide copy for slides -- the same figure as nmdb_n2p.py
            plot_paths = list(plot_paths) + n2p.make_test_diagnostic_plots(
                fig_dir, list(day[test_idx]), true_by["test"], pred_by["test"],
                y_err[test_idx], e_bins,
                relerr_daily=y_err[test_idx] / y_raw[test_idx],
                relerr_label=POS_ERR_LABEL)
        except Exception as exc:
            print(f"[skip test diagnostic plots] {exc}")

    summary = {
        "framework": "pytorch", "torch_version": torch.__version__, "device": str(device),
        "proton_input": str(proton_meta),
        "proton_input_rule": "AMS measured where available (per day and bin), "
                             "n2p prediction elsewhere",
        "positron_target": str(args.positron_root),
        "positron_bins_GeV": e_bins, "proton_bins_GV": p_bins,
        "aligned_days": int(n_days),
        "aligned_range": [first_day.isoformat(), last_day.isoformat()],
        "avg_window": None if win is None else {
            "scale": args.avg_window, "length_days": int(agg.SCALES[args.avg_window]),
            "n_windows": int(n_days), "n_windows_with_data": int(win["n_windows_with_data"]),
            "label": "exposure-weighted window mean (hexps-pos.root), "
                     "err sqrt(sum((T e)^2))/sum(T)",
            "input": "equal-weight mean of the proton input over the window days "
                     "with a positron value",
            "exposure": str(args.exposure_root),
            "daily_prediction": daily},
        "proton_input_fully_measured_days": int((meas_frac == 1).sum()),
        "proton_input_fully_predicted_days": int((meas_frac == 0).sum()),
        "proton_input_out_of_range_days": int(oor.sum()),
        "kernels": kernels, "trainable_parameters": int(n_params),
        "data_seed": args.data_seed, "epochs": args.epochs,
        "epochs_run": len(train_losses), "best_epoch": best_epoch,
        "best_val_loss": float(best_val) if best_epoch > 0 else None,
        "init": args.init, "drop_last": args.drop_last,
        "relerr_ylim": args.relerr_ylim,
        "train_range": args.train_range,
        "train_days": ("days inside the AMS PRL published range only: PRL measured "
                       "proton where present (per day and bin), n2p interpolation in "
                       "the gaps" if args.train_range == "ams" else
                       "all aligned days (AMS-measured proton where available, "
                       "n2p prediction elsewhere, incl. out-of-range inputs)"),
        # with --loss-weight err the losses (and best_val_loss) are chi^2 / c,
        # not comparable to plain-MSE runs; compare the metrics instead
        "loss_weight": None if not use_w else {
            "type": "err", "error": "posflux.root hflux_2d errors / flux",
            "formula": "w = (R_k*ln10/eps_ik)^2 / c, c = train mean",
            "c": loss_weight_c, "n_eps_filled": n_eps_filled},
        "batch_size": args.batch_size, "learning_rate": args.lr, "l2": args.l2,
        "dropout": args.dropout, "test_frac": args.test_frac, "val_frac": args.val_frac,
        "early_stopping": {"patience": args.patience, "min_delta": args.min_delta},
        "metrics": metrics,
        # chi2dof_total here uses the posflux.root error (no daily/total split)
        "paper_metrics": paper,
        "outputs": {"checkpoint": str(ckpt), "scalers": str(model_dir / "scalers.npz"),
                    "figure_dir": str(fig_dir), "error_dir": str(err_dir),
                    "plots": plot_paths, "timeseries_plots": ts_paths},
    }
    (out_dir / "p2pos_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"summary: {out_dir / 'p2pos_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
