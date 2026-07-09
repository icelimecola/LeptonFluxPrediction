#!/bin/python

import argparse
import subprocess
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from common import ELECTRON_DIR, FLUX_START, ensure_dirs, flux_figure_dir, imputation_path, interpolate_nan_by_bin


sys.path.insert(0, str(ELECTRON_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate conservative errors for sun-imputed electron flux.")
    parser.add_argument("--err", type=str, default=str(imputation_path("electron_err_observed_nan.npy")))
    parser.add_argument("--mask", type=str, default=str(imputation_path("electron_observed_mask.npy")))
    parser.add_argument("--pred-flux", type=str, default=str(imputation_path("electron_flux_sun_pred_allbin.npy")))
    parser.add_argument("--rrmse", type=str, default=str(imputation_path("sun_imputer_validation_relative_rmse_per_bin.npy")))
    parser.add_argument("--default-rrmse", type=float, default=0.3)
    parser.add_argument("--no-plots", action="store_true", help="Only save npy files; skip pdf/html overview plots.")
    parser.add_argument("--no-eval-plots", action="store_true", help="Skip lstmdraw-style first-stage evaluation plots.")
    return parser.parse_args()


def load_dates(n_days):
    date_path = imputation_path("electron_observed_dates.csv")
    if date_path.exists():
        dates = pd.read_csv(date_path)["date"]
        return pd.to_datetime(dates).iloc[:n_days]
    return pd.date_range(start=FLUX_START, periods=n_days, freq="D")


def load_energy_edges(n_bins):
    edge_path = imputation_path("electron_energy_edges.npy")
    if edge_path.exists():
        edges = np.load(edge_path)
        if len(edges) == n_bins + 1:
            return edges
    fallback_path = ELECTRON_DIR / "step2" / "Data" / "flux" / "electron_energy_edges.npy"
    if fallback_path.exists():
        edges = np.load(fallback_path)
        if len(edges) == n_bins + 1:
            return edges
    return np.arange(n_bins + 1, dtype=float)


def bin_title(edges, centers, bi):
    return f"{centers[bi]:.2f} GeV  [{edges[bi]:.2f}-{edges[bi + 1]:.2f}] GeV"


def write_flux_html(output_path, dates, flux, err, mask_obs, edges, trace_mode="lines", with_error=False):
    from tool_plotlyhtml import make_trace, write_plotly_panels

    centers = 0.5 * (edges[:-1] + edges[1:])
    targets = [1.0, 2.0, 5.0, 10.0]
    plot_bins = [int(np.argmin(np.abs(centers - target))) for target in targets]

    panels = []
    for j, bi in enumerate(plot_bins):
        title = bin_title(edges, centers, bi)
        observed = mask_obs[:, bi].astype(bool)
        missing = ~observed
        flux_obs = np.where(observed, flux[:, bi], np.nan)
        flux_imp = np.where(missing, flux[:, bi], np.nan)
        err_obs = np.where(observed, err[:, bi], np.nan)
        err_imp = np.where(missing, err[:, bi], np.nan)
        traces = []

        if trace_mode == "lines":
            traces.append(
                make_trace(
                    "final flux",
                    dates,
                    flux[:, bi],
                    mode="lines",
                    color="#1f77b4",
                    width=0.8,
                    hovertemplate=(
                        f"{title}<br>date=%{{x|%Y-%m-%d}}<br>"
                        "flux=%{y:.6g}<extra></extra>"
                    ),
                    showlegend=(j == 0),
                )
            )

        common_hover = f"{title}<br>date=%{{x|%Y-%m-%d}}<br>flux=%{{y:.6g}}"
        if with_error:
            common_hover += "<br>error=%{customdata:.6g}"
        common_hover += "<extra></extra>"

        traces.append(
            make_trace(
                "observed",
                dates,
                flux_obs,
                mode="markers" if trace_mode == "markers" else "markers",
                color="#1f77b4",
                marker_size=3,
                error_y=err_obs if with_error else None,
                customdata=err_obs if with_error else None,
                hovertemplate=common_hover,
                showlegend=(j == 0),
            )
        )
        traces.append(
            make_trace(
                "sun-imputed",
                dates,
                flux_imp,
                mode="markers",
                color="#d62728",
                marker_size=4,
                error_y=err_imp if with_error else None,
                customdata=err_imp if with_error else None,
                hovertemplate=common_hover,
                showlegend=(j == 0),
            )
        )
        panels.append({"title": title, "traces": traces})

    write_plotly_panels(
        output_path,
        "Electron Sun-Imputed Flux Overview",
        panels,
        columns=2,
        yaxis_title="Electron Flux [m^-2 s^-1 sr^-1 (GeV/n)^-1]",
    )


def write_flux_plots(dates, flux, err, mask_obs):
    import matplotlib.pyplot as plt

    out_dir = flux_figure_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = load_energy_edges(flux.shape[1])
    centers = 0.5 * (edges[:-1] + edges[1:])
    targets = [1.0, 2.0, 5.0, 10.0]
    plot_bins = [int(np.argmin(np.abs(centers - target))) for target in targets]

    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10

    fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    for j, bi in enumerate(plot_bins):
        ax = axs[j // 2, j % 2]
        observed = mask_obs[:, bi].astype(bool)
        missing = ~observed
        ax.plot(dates, flux[:, bi], color="tab:blue", lw=0.5, label="final flux")
        ax.plot(dates[observed], flux[observed, bi], ".", color="tab:blue", markersize=1.5, label="observed")
        ax.plot(dates[missing], flux[missing, bi], ".", color="tab:red", markersize=2.2, label="sun-imputed")
        ax.set_title(bin_title(edges, centers, bi), fontsize=11)
        ax.set_ylabel("Flux")
        if j == 0:
            ax.legend(loc="best", fontsize=9)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.08, wspace=0.08, hspace=0.15)
    fig.supxlabel("Year", y=0.03, fontsize=13)
    fig.supylabel("Electron Flux  [m$^{-2}$ s$^{-1}$ sr$^{-1}$ (GeV/n)$^{-1}$]", x=0.03, fontsize=13)
    plt.savefig(out_dir / "electron_flux_sun_imputed_overview.pdf", bbox_inches="tight")
    plt.close()
    write_flux_html(out_dir / "electron_flux_sun_imputed_overview.html", dates, flux, err, mask_obs, edges, "lines", False)

    fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    for j, bi in enumerate(plot_bins):
        ax = axs[j // 2, j % 2]
        observed = mask_obs[:, bi].astype(bool)
        missing = ~observed
        ax.plot(dates[observed], flux[observed, bi], ".", color="tab:blue", markersize=1.5, label="observed")
        ax.plot(dates[missing], flux[missing, bi], ".", color="tab:red", markersize=2.2, label="sun-imputed")
        ax.set_title(bin_title(edges, centers, bi), fontsize=11)
        ax.set_ylabel("Flux")
        if j == 0:
            ax.legend(loc="best", fontsize=9)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.08, wspace=0.08, hspace=0.15)
    fig.supxlabel("Year", y=0.03, fontsize=13)
    fig.supylabel("Electron Flux  [m$^{-2}$ s$^{-1}$ sr$^{-1}$ (GeV/n)$^{-1}$]", x=0.03, fontsize=13)
    plt.savefig(out_dir / "electron_flux_sun_imputed_overview_points.pdf", bbox_inches="tight")
    plt.close()
    write_flux_html(out_dir / "electron_flux_sun_imputed_overview_points.html", dates, flux, err, mask_obs, edges, "markers", False)

    fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    for j, bi in enumerate(plot_bins):
        ax = axs[j // 2, j % 2]
        observed = mask_obs[:, bi].astype(bool)
        missing = ~observed
        ax.errorbar(
            dates[observed],
            flux[observed, bi],
            yerr=err[observed, bi],
            fmt=".",
            markersize=1.5,
            elinewidth=0.45,
            capsize=0,
            color="tab:blue",
            ecolor="tab:blue",
            alpha=0.35,
            label="observed",
        )
        ax.errorbar(
            dates[missing],
            flux[missing, bi],
            yerr=err[missing, bi],
            fmt=".",
            markersize=2.2,
            elinewidth=0.55,
            capsize=0,
            color="tab:red",
            ecolor="tab:red",
            alpha=0.45,
            label="sun-imputed",
        )
        ax.set_title(bin_title(edges, centers, bi), fontsize=11)
        ax.set_ylabel("Flux")
        if j == 0:
            ax.legend(loc="best", fontsize=9)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.08, wspace=0.08, hspace=0.15)
    fig.supxlabel("Year", y=0.03, fontsize=13)
    fig.supylabel("Electron Flux  [m$^{-2}$ s$^{-1}$ sr$^{-1}$ (GeV/n)$^{-1}$]", x=0.03, fontsize=13)
    plt.savefig(out_dir / "electron_flux_sun_imputed_overview_with_error.pdf", bbox_inches="tight")
    plt.close()
    write_flux_html(out_dir / "electron_flux_sun_imputed_overview_with_error.html", dates, flux, err, mask_obs, edges, "markers", True)

    print("Saved plots to:", out_dir)


def main():
    ensure_dirs()
    args = parse_args()

    err_obs = np.load(args.err)
    mask_obs = np.load(args.mask).astype(bool)
    pred_flux = np.load(args.pred_flux)
    if pred_flux.shape != err_obs.shape:
        raise ValueError(f"pred_flux shape {pred_flux.shape} does not match err shape {err_obs.shape}")

    rrmse_path = Path(args.rrmse)
    if args.rrmse and rrmse_path.exists():
        rrmse = np.load(args.rrmse)
    else:
        rrmse = np.full(err_obs.shape[1], args.default_rrmse, dtype=float)
    rrmse = np.where(np.isfinite(rrmse) & (rrmse > 0), rrmse, args.default_rrmse)

    neighbor_err = interpolate_nan_by_bin(err_obs)
    model_err = np.maximum(pred_flux, 0.0) * rrmse.reshape(1, -1)
    err_imputed = err_obs.copy()
    err_imputed[~mask_obs] = np.maximum(neighbor_err[~mask_obs], model_err[~mask_obs])
    err_imputed[mask_obs] = err_obs[mask_obs]
    err_imputed = np.where(np.isfinite(err_imputed) & (err_imputed >= 0), err_imputed, neighbor_err)

    np.save(imputation_path("electron_err_neighbor_interp.npy"), neighbor_err)
    np.save(imputation_path("electron_err_sun_imputed.npy"), err_imputed)

    print("Saved neighbor/interpolated err:", imputation_path("electron_err_neighbor_interp.npy"))
    print("Saved imputed err:", imputation_path("electron_err_sun_imputed.npy"))
    print("Remaining NaN:", int(np.sum(~np.isfinite(err_imputed))))

    if not args.no_plots:
        flux_imputed_path = imputation_path("electron_flux_sun_imputed.npy")
        if not flux_imputed_path.exists():
            raise FileNotFoundError(f"Missing final imputed flux file: {flux_imputed_path}")
        flux_imputed = np.load(flux_imputed_path)
        if flux_imputed.shape != err_imputed.shape:
            raise ValueError(f"flux shape {flux_imputed.shape} does not match err shape {err_imputed.shape}")
        dates = load_dates(flux_imputed.shape[0])
        write_flux_plots(dates, flux_imputed, err_imputed, mask_obs)

    if not args.no_eval_plots:
        subprocess.run([sys.executable, str(Path(__file__).with_name("draw_sun_imputer.py"))], check=True)


if __name__ == "__main__":
    main()
