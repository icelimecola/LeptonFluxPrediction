#!/usr/bin/env python3
"""Train a proton -> positron residual CNN (same network family as nmdb_n2p.py).

Input  X : daily proton flux, 30 rigidity bins, 2001-2026, from
           nmdb_n2p_extend.py (data/nmdb_n2p_extend_<G>/proton_extend_meta.npz).
           Spliced per (day, bin): the AMS measurement where it exists
           (flux_ams), otherwise the n2p prediction (flux_pred).
Target y : AMS daily positron flux from rawdata/flux/posflux.root (hflux_2d),
           processed like POS/step2/dataproc_fluxpos.py -- first and last
           energy bins dropped (27 bins, 1.0-41.9 GeV), all-zero head/tail
           days trimmed, remaining zeros = missing -- but NOT interpolated:
           only days with every kept bin measured are used.

Training mirrors nmdb_n2p.py: log10 on both sides, MinMax fitted on the train
rows, random 80/10/10 split by --data-seed, Adamax, MSE + Keras-style L2 on
the output kernel, Keras BatchNorm defaults, early stopping on val loss, best
weights restored. The convolution runs along the proton rigidity axis.

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
  p2pos_summary.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path

import numpy as np

import nmdb_n2p as n2p

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
                        default=BASE.parent / "rawdata" / "flux" / "posflux.root")
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

    # training days: every positron bin measured, and inside the proton range
    lookup = {d: i for i, d in enumerate(p_dates)}
    complete = np.isfinite(e_flux).all(axis=1)
    pos_rows = [k for k, d in enumerate(e_dates) if complete[k] and d in lookup]
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
    n_days, n_in = X_raw.shape
    n_out = y_raw.shape[1]
    print(f"aligned days={n_days}  proton bins={n_in}  positron bins={n_out}")
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

    model, kernels = n2p.build_model(n_in, n_out, args.dropout, init=args.init)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"kernels: {kernels}  trainable parameters: {n_params:,}")
    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.999), eps=1e-7)
    mse = torch.nn.MSELoss()

    def criterion(pred, target):   # Keras L2(l2) on the output kernel only
        return mse(pred, target) + args.l2 * torch.sum(model.head.weight ** 2)

    def tensor(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    Xv, yv = tensor(sets["val"][0]), tensor(sets["val"][1])
    Xt, yt = tensor(sets["test"][0]), tensor(sets["test"][1])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.as_tensor(sets["train"][0], dtype=torch.float32),
            torch.as_tensor(sets["train"][1], dtype=torch.float32)),
        batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last)

    model_dir, fig_dir, err_dir = out_dir / "Model", out_dir / "Figure", out_dir / "Error"
    for path in (model_dir, fig_dir, err_dir):
        path.mkdir(parents=True, exist_ok=True)
    tag = (f"seed{args.data_seed}_p{n_in}e{n_out}_{args.epochs}ep_lr{args.lr}"
           f"_l2{args.l2}_do{args.dropout}_b{args.batch_size}")
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
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(Xv), yv).item())
            test_loss = float(criterion(model(Xt), yt).item())
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
    np.savez(err_dir / f"{tag}_test_meta.npz",
             dates=np.asarray([d.isoformat() for d in day[test_idx]]),
             proton_measured_fraction=meas_frac[test_idx],
             proton_input_out_of_range=oor[test_idx],
             # for nmdb_n2p_replot_relerr.py: redraw without retraining
             positron_rel_err=y_err[test_idx] / y_raw[test_idx],
             energy_min=np.asarray([b[0] for b in e_bins]),
             energy_max=np.asarray([b[1] for b in e_bins]))

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
        "aligned_range": [day[0].isoformat(), day[-1].isoformat()],
        "proton_input_fully_measured_days": int((meas_frac == 1).sum()),
        "proton_input_fully_predicted_days": int((meas_frac == 0).sum()),
        "proton_input_out_of_range_days": int(oor.sum()),
        "kernels": kernels, "trainable_parameters": int(n_params),
        "data_seed": args.data_seed, "epochs": args.epochs,
        "epochs_run": len(train_losses), "best_epoch": best_epoch,
        "best_val_loss": float(best_val) if best_epoch > 0 else None,
        "init": args.init, "drop_last": args.drop_last,
        "relerr_ylim": args.relerr_ylim,
        "train_days": "all aligned days (AMS-measured proton where available, "
                      "n2p prediction elsewhere, incl. out-of-range inputs)",
        "batch_size": args.batch_size, "learning_rate": args.lr, "l2": args.l2,
        "dropout": args.dropout, "test_frac": args.test_frac, "val_frac": args.val_frac,
        "early_stopping": {"patience": args.patience, "min_delta": args.min_delta},
        "metrics": metrics,
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
