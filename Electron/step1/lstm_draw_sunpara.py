#!/bin/python

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    ELECTRON_DIR,
    FLUX_START,
    PAD_DAYS,
    SUN_OFFSET,
    draw_data_path,
    draw_figure_dir,
    ensure_dirs,
    find_best_model,
    flux_figure_dir,
    imputation_path,
    interpolate_nan_by_bin,
    load_sun_daily,
    minmax_inverse,
    minmax_transform,
)


sys.path.insert(0, str(ELECTRON_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Predict missing electron flux and build final step1 flux/err products.")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--flux", type=str, default=str(imputation_path("electron_flux_observed_nan.npy")))
    parser.add_argument("--mask", type=str, default=str(imputation_path("electron_observed_mask.npy")))
    parser.add_argument("--err", type=str, default=str(imputation_path("electron_err_observed_nan.npy")))
    parser.add_argument("--rrmse", type=str, default=str(imputation_path("sun_imputer_validation_relative_rmse_per_bin.npy")))
    parser.add_argument("--default-rrmse", type=float, default=0.3)
    parser.add_argument("--no-plots", action="store_true", help="Skip final flux overview plots.")
    parser.add_argument("--no-eval-plots", action="store_true", help="Skip lstmdraw-style first-stage evaluation plots.")
    return parser.parse_args()


def make_sequence(X, look_back):
    return np.array([X[i - look_back:i, :] for i in range(look_back, len(X))])


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
                mode="markers",
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


def estimate_imputed_err(err_obs, mask_obs, pred_flux, rrmse_path, default_rrmse):
    rrmse_path = Path(rrmse_path)
    if rrmse_path.exists():
        rrmse = np.load(rrmse_path)
    else:
        rrmse = np.full(err_obs.shape[1], default_rrmse, dtype=float)
    rrmse = np.where(np.isfinite(rrmse) & (rrmse > 0), rrmse, default_rrmse)

    neighbor_err = interpolate_nan_by_bin(err_obs)
    model_err = np.maximum(pred_flux, 0.0) * rrmse.reshape(1, -1)
    err_imputed = err_obs.copy()
    err_imputed[~mask_obs] = np.maximum(neighbor_err[~mask_obs], model_err[~mask_obs])
    err_imputed[mask_obs] = err_obs[mask_obs]
    err_imputed = np.where(np.isfinite(err_imputed) & (err_imputed >= 0), err_imputed, neighbor_err)
    return neighbor_err, err_imputed


def model_tag(model_name):
    if not model_name:
        return "best"
    return model_name.rsplit("/", 1)[-1].replace(".keras", "")


def split_slices(n_days, look_back, train_num, val_num):
    number = PAD_DAYS + n_days
    train_end = int(number * train_num)
    val_end = int(number * (train_num + val_num))
    train_start = max(0, look_back - PAD_DAYS)
    train_stop = min(n_days, max(0, train_end - PAD_DAYS))
    val_start = train_stop
    val_stop = min(n_days, max(0, val_end - PAD_DAYS))
    test_start = val_stop
    test_stop = n_days
    return {
        "train": slice(train_start, train_stop),
        "val": slice(val_start, val_stop),
        "test": slice(test_start, test_stop),
        "train_end": train_stop,
        "val_end": val_stop,
    }


def safe_relative_error(pred, true, mask):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    mask = np.asarray(mask).astype(bool)
    out = np.full_like(pred, np.nan, dtype=float)
    valid = mask & np.isfinite(pred) & np.isfinite(true) & (true > 0)
    out[valid] = pred[valid] / true[valid] - 1.0
    return out


def mean_abs_relative_error(err):
    return np.nanmean(np.abs(err))


def select_plot_bins(edges):
    centers = 0.5 * (edges[:-1] + edges[1:])
    targets = [1.0, 2.0, 5.0, 10.0]
    plot_bins = [int(np.argmin(np.abs(centers - target))) for target in targets]
    labels = [
        f"{centers[bi]:.2f} GeV  [{edges[bi]:.2f}-{edges[bi + 1]:.2f}]"
        for bi in plot_bins
    ]
    return plot_bins, labels


def masked_values(values, mask):
    return np.where(mask, values, np.nan)


def write_prediction_eval_html(output_path, dates, plot_bins, bin_labels, truth, err, mask, pred, splits):
    from tool_plotlyhtml import make_trace, write_plotly_panels

    panels = []
    segments = [
        ("training", splits["train"], "blue"),
        ("validation", splits["val"], "goldenrod"),
        ("test", splits["test"], "magenta"),
    ]
    for j, bi in enumerate(plot_bins):
        title = bin_labels[j]
        traces = []
        for segment_name, seg, _ in segments:
            seg_mask = mask[seg, bi]
            traces.append(
                make_trace(
                    f"{segment_name} observation",
                    dates[seg],
                    masked_values(truth[seg, bi], seg_mask),
                    mode="markers",
                    color="green",
                    marker_size=4,
                    error_y=masked_values(err[seg, bi], seg_mask),
                    customdata=masked_values(err[seg, bi], seg_mask),
                    hovertemplate=(
                        f"{title}<br>"
                        "date=%{x|%Y-%m-%d}<br>"
                        "observation=%{y:.6g}<br>"
                        "error=%{customdata:.6g}<extra></extra>"
                    ),
                    showlegend=(j == 0),
                )
            )
        for segment_name, seg, color in segments:
            traces.append(
                make_trace(
                    f"{segment_name} prediction",
                    dates[seg],
                    pred[seg, bi],
                    mode="lines",
                    color=color,
                    width=1.0,
                    hovertemplate=(
                        f"{title}<br>"
                        "date=%{x|%Y-%m-%d}<br>"
                        f"{segment_name} prediction=%{{y:.6g}}<extra></extra>"
                    ),
                    showlegend=(j == 0),
                )
            )
        panels.append({
            "title": title,
            "traces": traces,
            "vlines": [dates.iloc[i] for i in [splits["train_end"], splits["val_end"]] if 0 <= i < len(dates)],
        })

    write_plotly_panels(
        output_path,
        "Sun-Only Imputer Electron Flux Prediction",
        panels,
        columns=2,
        yaxis_title="Electron Flux",
    )


def write_error_eval_html(output_path, dates, plot_bins, bin_labels, errors, splits):
    from tool_plotlyhtml import make_trace, write_plotly_panels

    panels = []
    segments = [
        ("training error", splits["train"], errors["train"], "blue"),
        ("validation error", splits["val"], errors["val"], "goldenrod"),
        ("test error", splits["test"], errors["test"], "magenta"),
    ]
    for j, bi in enumerate(plot_bins):
        title = bin_labels[j]
        traces = []
        for name, seg, segment_error, color in segments:
            traces.append(
                make_trace(
                    name,
                    dates[seg],
                    segment_error[:, bi],
                    mode="lines",
                    color=color,
                    width=1.0,
                    hovertemplate=(
                        f"{title}<br>"
                        "date=%{x|%Y-%m-%d}<br>"
                        f"{name}=%{{y:.6g}}<extra></extra>"
                    ),
                    showlegend=(j == 0),
                )
            )
        panels.append({
            "title": title,
            "traces": traces,
            "vlines": [dates.iloc[i] for i in [splits["train_end"], splits["val_end"]] if 0 <= i < len(dates)],
            "hlines": [0],
        })

    write_plotly_panels(
        output_path,
        "Sun-Only Imputer Relative Error",
        panels,
        columns=2,
        yaxis_title="Relative Error",
    )


def draw_pdf_prediction_eval(output_path, dates, plot_bins, bin_labels, truth, err, mask, pred, splits):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    segments = [
        ("training", splits["train"], "b"),
        ("validation", splits["val"], "y"),
        ("test", splits["test"], "m"),
    ]
    split_style = dict(color="0.3", ls=":", lw=1.0, alpha=0.7, zorder=0)

    for j, bi in enumerate(plot_bins):
        ax = axs[j // 2, j % 2]
        for _, seg, _ in segments:
            seg_mask = mask[seg, bi]
            ax.errorbar(
                dates[seg][seg_mask],
                truth[seg, bi][seg_mask],
                yerr=err[seg, bi][seg_mask],
                fmt=".",
                color="g",
                ecolor="g",
                markersize=1.8,
                elinewidth=0.45,
                capsize=0,
                alpha=0.65,
                zorder=1,
            )
        for _, seg, color in segments:
            ax.plot(dates[seg], pred[seg, bi], color + "-", lw=0.5)
        ax.text(0.02, 0.88, bin_labels[j], transform=ax.transAxes, fontsize=11, fontweight="bold")
        for split in [splits["train_end"], splits["val_end"]]:
            if 0 <= split < len(dates):
                ax.axvline(dates.iloc[split], **split_style)

        if j >= 2:
            trans = ax.get_xaxis_transform()
            for label, seg, _ in segments:
                if seg.stop > seg.start:
                    mid = int((seg.start + seg.stop) / 2)
                    ax.text(dates.iloc[mid], 0.03, label, transform=trans, ha="center", va="bottom", fontsize=10)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.96, bottom=0.08, wspace=0.06, hspace=0.10)
    fig.supylabel("Electron Flux  [m$^{-2}$ s$^{-1}$ sr$^{-1}$ (GeV/n)$^{-1}$]", x=0.03, fontsize=14)
    fig.supxlabel("Year", y=0.03, fontsize=14)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def draw_pdf_error_eval(output_path, dates, plot_bins, bin_labels, errors, splits):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    segments = [
        ("train", splits["train"], errors["train"], "b"),
        ("val", splits["val"], errors["val"], "y"),
        ("test", splits["test"], errors["test"], "m"),
    ]
    split_style = dict(color="0.3", ls=":", lw=1.0, alpha=0.7, zorder=0)

    for j, bi in enumerate(plot_bins):
        ax = axs[j // 2, j % 2]
        for _, seg, segment_error, color in segments:
            ax.plot(dates[seg], segment_error[:, bi], color + "-", lw=0.5)
        ax.axhline(0, color="k", lw=0.5)
        ax.text(0.02, 0.88, bin_labels[j], transform=ax.transAxes, fontsize=11, fontweight="bold")
        for split in [splits["train_end"], splits["val_end"]]:
            if 0 <= split < len(dates):
                ax.axvline(dates.iloc[split], **split_style)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.96, bottom=0.08, wspace=0.06, hspace=0.10)
    fig.supylabel("Relative Error", x=0.03, fontsize=14)
    fig.supxlabel("Year", y=0.03, fontsize=14)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def write_eval_plots(truth, err, mask, pred, scalers, model_name):
    look_back = int(scalers["look_back"])
    train_num = float(scalers["train_num"])
    val_num = float(scalers["val_num"])
    tag = model_tag(model_name)

    dates = pd.Series(load_dates(truth.shape[0]))
    edges = load_energy_edges(truth.shape[1])
    plot_bins, bin_labels = select_plot_bins(edges)
    splits = split_slices(truth.shape[0], look_back, train_num, val_num)

    train_error = safe_relative_error(pred[splits["train"]], truth[splits["train"]], mask[splits["train"]])
    val_error = safe_relative_error(pred[splits["val"]], truth[splits["val"]], mask[splits["val"]])
    test_error = safe_relative_error(pred[splits["test"]], truth[splits["test"]], mask[splits["test"]])
    errors = {"train": train_error, "val": val_error, "test": test_error}

    np.save(draw_data_path("train_error_allbin_" + tag + ".npy"), train_error)
    np.save(draw_data_path("val_error_allbin_" + tag + ".npy"), val_error)
    np.save(draw_data_path("test_error_allbin_" + tag + ".npy"), test_error)

    print(
        "mean absolute relative error:",
        "training", mean_abs_relative_error(train_error),
        "validation", mean_abs_relative_error(val_error),
        "test", mean_abs_relative_error(test_error),
    )

    out_dir = draw_figure_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_pdf = out_dir / ("electron_sun_imputer_prediction_" + tag + ".pdf")
    prediction_html = out_dir / ("electron_sun_imputer_prediction_" + tag + ".html")
    error_pdf = out_dir / ("electron_sun_imputer_error_" + tag + ".pdf")
    error_html = out_dir / ("electron_sun_imputer_error_" + tag + ".html")

    draw_pdf_prediction_eval(prediction_pdf, dates, plot_bins, bin_labels, truth, err, mask, pred, splits)
    write_prediction_eval_html(prediction_html, dates, plot_bins, bin_labels, truth, err, mask, pred, splits)
    draw_pdf_error_eval(error_pdf, dates, plot_bins, bin_labels, errors, splits)
    write_error_eval_html(error_html, dates, plot_bins, bin_labels, errors, splits)

    print("Saved first-stage prediction plot:", prediction_pdf)
    print("Saved first-stage error plot:", error_pdf)


def main():
    ensure_dirs()
    args = parse_args()

    import tensorflow as tf
    from tensorflow.keras.models import load_model

    def masked_huber(y_packed, y_pred, delta=1.0):
        bins = y_pred.shape[-1]
        y_true = y_packed[:, :bins]
        mask = y_packed[:, bins:]
        abs_error = tf.abs(y_true - y_pred)
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic
        loss = 0.5 * tf.square(quadratic) + delta * linear
        denom = tf.maximum(tf.reduce_sum(mask), 1.0)
        return tf.reduce_sum(loss * mask) / denom

    flux_obs = np.load(args.flux)
    err_obs = np.load(args.err)
    mask_obs = np.load(args.mask).astype(bool)
    scalers = np.load(imputation_path("sun_imputer_scalers.npz"))
    look_back = int(scalers["look_back"])
    x_min, x_max = scalers["x_min"], scalers["x_max"]
    y_min, y_max = scalers["y_min"], scalers["y_max"]

    model_path = args.model if args.model else str(find_best_model("sunImputer_"))
    model = load_model(model_path, custom_objects={"masked_huber": masked_huber})

    bins = flux_obs.shape[1]
    flux_padded = np.concatenate([np.full((PAD_DAYS, bins), np.nan), flux_obs], axis=0)
    sun_daily = load_sun_daily()
    X_all = sun_daily[SUN_OFFSET:SUN_OFFSET + len(flux_padded), 0:5]
    X_all = minmax_transform(X_all, x_min, x_max)
    X_seq = make_sequence(X_all, look_back)

    pred_scaled = model.predict(X_seq, verbose=1)
    pred_all_targets = minmax_inverse(pred_scaled, y_min, y_max)

    target_indices = np.arange(look_back, len(flux_padded))
    actual_rows = target_indices - PAD_DAYS
    keep = (actual_rows >= 0) & (actual_rows < len(flux_obs))
    pred_flux = pred_all_targets[keep]
    if pred_flux.shape != flux_obs.shape:
        raise ValueError(f"Prediction shape {pred_flux.shape} does not match observed flux shape {flux_obs.shape}")
    if err_obs.shape != flux_obs.shape:
        raise ValueError(f"Error shape {err_obs.shape} does not match observed flux shape {flux_obs.shape}")
    if mask_obs.shape != flux_obs.shape:
        raise ValueError(f"Mask shape {mask_obs.shape} does not match observed flux shape {flux_obs.shape}")

    flux_imputed = flux_obs.copy()
    flux_imputed[~mask_obs] = pred_flux[~mask_obs]

    neighbor_err, err_imputed = estimate_imputed_err(
        err_obs,
        mask_obs,
        pred_flux,
        args.rrmse,
        args.default_rrmse,
    )

    np.save(imputation_path("electron_flux_sun_pred_allbin.npy"), pred_flux)
    np.save(imputation_path("electron_flux_sun_imputed.npy"), flux_imputed)
    np.save(imputation_path("electron_err_neighbor_interp.npy"), neighbor_err)
    np.save(imputation_path("electron_err_sun_imputed.npy"), err_imputed)

    print("Loaded model:", model_path)
    print("Saved all sun-only predictions:", imputation_path("electron_flux_sun_pred_allbin.npy"))
    print("Saved imputed flux:", imputation_path("electron_flux_sun_imputed.npy"))
    print("Saved neighbor/interpolated err:", imputation_path("electron_err_neighbor_interp.npy"))
    print("Saved imputed err:", imputation_path("electron_err_sun_imputed.npy"))
    print("Remaining flux NaN:", int(np.sum(~np.isfinite(flux_imputed))))
    print("Remaining err NaN:", int(np.sum(~np.isfinite(err_imputed))))

    if not args.no_plots:
        dates = load_dates(flux_imputed.shape[0])
        write_flux_plots(dates, flux_imputed, err_imputed, mask_obs)

    if not args.no_eval_plots:
        write_eval_plots(flux_obs, err_imputed, mask_obs, pred_flux, scalers, Path(model_path).name)


if __name__ == "__main__":
    main()
