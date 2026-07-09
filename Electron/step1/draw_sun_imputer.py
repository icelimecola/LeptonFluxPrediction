#!/bin/python
# -*- coding: utf-8 -*-

import argparse
import sys

import numpy as np
import pandas as pd

from common import (
    ELECTRON_DIR,
    FLUX_START,
    PAD_DAYS,
    draw_data_path,
    draw_figure_dir,
    ensure_dirs,
    find_best_model,
    imputation_path,
)


sys.path.insert(0, str(ELECTRON_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Draw lstmdraw-style evaluation plots for the sun-only imputer.")
    parser.add_argument("--flux-observed", type=str, default=str(imputation_path("electron_flux_observed_nan.npy")))
    parser.add_argument("--err", type=str, default=str(imputation_path("electron_err_sun_imputed.npy")))
    parser.add_argument("--mask", type=str, default=str(imputation_path("electron_observed_mask.npy")))
    parser.add_argument("--pred-flux", type=str, default=str(imputation_path("electron_flux_sun_pred_allbin.npy")))
    parser.add_argument("--scalers", type=str, default=str(imputation_path("sun_imputer_scalers.npz")))
    parser.add_argument("--model-name", type=str, default="")
    return parser.parse_args()


def load_dates(n_days):
    date_path = imputation_path("electron_observed_dates.csv")
    if date_path.exists():
        dates = pd.read_csv(date_path)["date"]
        return pd.to_datetime(dates).iloc[:n_days]
    return pd.Series(pd.date_range(start=FLUX_START, periods=n_days, freq="D"))


def load_energy_edges(n_bins):
    for path in [
        imputation_path("electron_energy_edges.npy"),
        ELECTRON_DIR / "step2" / "Data" / "flux" / "electron_energy_edges.npy",
    ]:
        if path.exists():
            edges = np.load(path)
            if len(edges) == n_bins + 1:
                return edges
    return np.arange(n_bins + 1, dtype=float)


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
        "look_back_actual": train_start,
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


def write_prediction_html(output_path, dates, plot_bins, bin_labels, truth, err, mask, pred, splits):
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


def write_error_html(output_path, dates, plot_bins, bin_labels, errors, splits):
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


def draw_pdf_prediction(output_path, dates, plot_bins, bin_labels, truth, err, mask, pred, splits):
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


def draw_pdf_error(output_path, dates, plot_bins, bin_labels, errors, splits):
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


def main():
    ensure_dirs()
    args = parse_args()

    truth = np.load(args.flux_observed)
    err = np.load(args.err)
    mask = np.load(args.mask).astype(bool)
    pred = np.load(args.pred_flux)
    scalers = np.load(args.scalers)

    if pred.shape != truth.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match truth shape {truth.shape}")
    if err.shape != truth.shape:
        raise ValueError(f"Error shape {err.shape} does not match truth shape {truth.shape}")
    if mask.shape != truth.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match truth shape {truth.shape}")

    look_back = int(scalers["look_back"])
    train_num = float(scalers["train_num"])
    val_num = float(scalers["val_num"])

    model_name = args.model_name
    if not model_name:
        model_name = find_best_model("sunImputer_").name
    tag = model_tag(model_name)

    dates = load_dates(truth.shape[0])
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

    draw_pdf_prediction(prediction_pdf, dates, plot_bins, bin_labels, truth, err, mask, pred, splits)
    write_prediction_html(prediction_html, dates, plot_bins, bin_labels, truth, err, mask, pred, splits)
    draw_pdf_error(error_pdf, dates, plot_bins, bin_labels, errors, splits)
    write_error_html(error_html, dates, plot_bins, bin_labels, errors, splits)

    print("Saved first-stage prediction plot:", prediction_pdf)
    print("Saved first-stage error plot:", error_pdf)


if __name__ == "__main__":
    main()
