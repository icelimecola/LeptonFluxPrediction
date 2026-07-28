#!/bin/python
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FLUX_PATH = Path('Data/flux/electron_flux_allbin.npy')
ERR_PATH = Path('Data/flux/electron_flux_abs_error_allbin.npy')
ENERGY_PATH = Path('Data/flux/electron_energy_edges.npy')
SUN_PATH = Path('../../sun_processed/latest/sun5_daily_all_latest.npy')
MODEL_SUMMARY_PATH = Path('Data/modelw/best_model.txt')
FIG_DIR = Path('Figure/fluxsunpara')

FLUX_START = pd.Timestamp('2011-06-11')
SUN_START = pd.Timestamp('2010-05-20')
SUN_LABELS = [
    'B (nT)',
    'Vsw (km/s)',
    'Tilt (degree)',
    'A',
    'SSN',
]
SUN_SHORT = ['B', 'Vsw', 'Tilt', 'A', 'SSN']
SUN_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
SPLIT_COLORS = {'train': 'tab:blue', 'validation': 'goldenrod', 'test': 'magenta'}


MODEL_SPLIT_RE = re.compile(
    r'_(?P<train_num>[-+0-9.eE]+)train_(?P<val_num>[-+0-9.eE]+)val_\d+-[-+0-9.eE]+\.keras$'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Draw electron flux vs solar-parameter lag diagnostic figures.'
    )
    parser.add_argument('--max-lag', type=int, default=720, help='Maximum lag in days.')
    parser.add_argument('--lag-step', type=int, default=1, help='Lag scan step in days.')
    parser.add_argument('--smooth-days', type=int, default=27, help='Rolling mean window in days.')
    parser.add_argument('--train-num', type=float, default=None, help='Train split fraction.')
    parser.add_argument('--val-num', type=float, default=None, help='Validation split fraction.')
    parser.add_argument('--min-points', type=int, default=30, help='Minimum valid points for correlation.')
    return parser.parse_args()


def parse_best_split():
    if not MODEL_SUMMARY_PATH.exists():
        return None
    model_name = MODEL_SUMMARY_PATH.read_text(encoding='utf-8').strip()
    match = MODEL_SPLIT_RE.search(model_name)
    if match is None:
        return None
    return float(match.group('train_num')), float(match.group('val_num'))


def moving_mean(values, window):
    if window <= 1:
        return np.asarray(values, dtype=float)
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window, center=True, min_periods=max(3, window // 3))
        .mean()
        .to_numpy()
    )


def zscore(values):
    arr = np.asarray(values, dtype=float)
    valid = np.isfinite(arr)
    out = np.full_like(arr, np.nan, dtype=float)
    if valid.sum() < 2:
        return out
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std == 0:
        return out
    out[valid] = (arr[valid] - mean) / std
    return out


def weighted_corr(x, y, weights=None, min_points=30):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        valid &= np.isfinite(w) & (w > 0)
        w = w[valid]
    else:
        w = None

    x = x[valid]
    y = y[valid]
    if x.size < min_points:
        return np.nan

    if w is None:
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    w_sum = np.sum(w)
    if w_sum <= 0:
        return np.nan
    mx = np.sum(w * x) / w_sum
    my = np.sum(w * y) / w_sum
    dx = x - mx
    dy = y - my
    vx = np.sum(w * dx * dx) / w_sum
    vy = np.sum(w * dy * dy) / w_sum
    if vx <= 0 or vy <= 0:
        return np.nan
    cov = np.sum(w * dx * dy) / w_sum
    return float(cov / np.sqrt(vx * vy))


def lagged_arrays(flux, solar, err, lag):
    if lag == 0:
        return flux, solar, err
    return flux[lag:], solar[:-lag], err[lag:]


def compute_lag_correlations(flux_smooth, sun_smooth, err, lags, min_points):
    n_bins = flux_smooth.shape[1]
    n_para = sun_smooth.shape[1]
    corr = np.full((n_bins, n_para, len(lags)), np.nan, dtype=float)
    wcorr = np.full_like(corr, np.nan)

    with np.errstate(divide='ignore', invalid='ignore'):
        weights = 1.0 / np.square(err)
    weights[~np.isfinite(weights)] = np.nan

    for bi in range(n_bins):
        for pi in range(n_para):
            for li, lag in enumerate(lags):
                y, x, e = lagged_arrays(flux_smooth[:, bi], sun_smooth[:, pi], err[:, bi], lag)
                _, _, w = lagged_arrays(flux_smooth[:, bi], sun_smooth[:, pi], weights[:, bi], lag)
                corr[bi, pi, li] = weighted_corr(x, y, None, min_points)
                wcorr[bi, pi, li] = weighted_corr(x, y, w, min_points)
    return corr, wcorr


def best_lag_summary(corr, wcorr, lags, plot_bins, bin_labels):
    rows = []
    for bi, label in zip(plot_bins, bin_labels):
        for pi, para in enumerate(SUN_SHORT):
            c = corr[bi, pi]
            wc = wcorr[bi, pi]
            best_i = int(np.nanargmax(np.abs(c))) if np.isfinite(c).any() else None
            best_wi = int(np.nanargmax(np.abs(wc))) if np.isfinite(wc).any() else None
            rows.append({
                'bin_index': bi,
                'energy_bin': label,
                'solar_parameter': para,
                'best_lag_days': np.nan if best_i is None else int(lags[best_i]),
                'best_corr': np.nan if best_i is None else float(c[best_i]),
                'best_abs_corr': np.nan if best_i is None else float(abs(c[best_i])),
                'weighted_best_lag_days': np.nan if best_wi is None else int(lags[best_wi]),
                'weighted_best_corr': np.nan if best_wi is None else float(wc[best_wi]),
                'weighted_best_abs_corr': np.nan if best_wi is None else float(abs(wc[best_wi])),
            })
    return pd.DataFrame(rows)


def split_info(n_days, train_num, val_num):
    train_end = int(n_days * train_num)
    val_end = int(n_days * (train_num + val_num))
    return train_end, val_end, n_days


def add_split_lines(ax, dates, train_end, val_end):
    style = dict(color='0.3', ls=':', lw=0.9, alpha=0.7)
    ax.axvline(dates[train_end], **style)
    ax.axvline(dates[val_end], **style)


def save_zscore_overlay(dates, flux_smooth, sun_smooth, plot_bins, bin_labels, train_end, val_end):
    fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    for j, (bi, label) in enumerate(zip(plot_bins, bin_labels)):
        ax = axs[j // 2, j % 2]
        ax.plot(dates, zscore(flux_smooth[:, bi]), color='black', lw=1.0, label='Flux')
        for pi, para in enumerate(SUN_SHORT):
            ax.plot(dates, zscore(sun_smooth[:, pi]), color=SUN_COLORS[pi], lw=0.8, alpha=0.85, label=para)
        add_split_lines(ax, dates, train_end, val_end)
        ax.axhline(0, color='0.6', lw=0.5)
        ax.set_title(label, fontsize=12, fontweight='bold')
        if j == 0:
            ax.legend(ncol=3, fontsize=9, loc='upper left')
    fig.suptitle('Z-score electron flux and solar parameters', fontsize=16)
    fig.supylabel('Z-score')
    fig.supxlabel('Year')
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(FIG_DIR / 'zscore_flux_sunpara_overlay.pdf', bbox_inches='tight')
    plt.close(fig)


def save_lag_corr(lags, corr_arr, plot_bins, bin_labels, suffix, title):
    fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True, sharey=True)
    for j, (bi, label) in enumerate(zip(plot_bins, bin_labels)):
        ax = axs[j // 2, j % 2]
        for pi, para in enumerate(SUN_SHORT):
            values = corr_arr[bi, pi]
            ax.plot(lags, values, color=SUN_COLORS[pi], lw=1.0, label=para)
            if np.isfinite(values).any():
                best_i = int(np.nanargmax(np.abs(values)))
                ax.plot(lags[best_i], values[best_i], 'o', color=SUN_COLORS[pi], ms=4)
        ax.axhline(0, color='0.4', lw=0.7)
        ax.set_title(label, fontsize=12, fontweight='bold')
        if j == 0:
            ax.legend(ncol=3, fontsize=9, loc='lower right')
    fig.suptitle(title, fontsize=16)
    fig.supylabel('Correlation coefficient')
    fig.supxlabel('Lag days: corr(flux(t), solar(t - lag))')
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(FIG_DIR / f'lag_correlation_{suffix}.pdf', bbox_inches='tight')
    plt.close(fig)


def save_faceted_timeseries(dates, flux, err, sun, flux_smooth, sun_smooth,
                            plot_bins, bin_labels, train_end, val_end):
    for bi, label in zip(plot_bins, bin_labels):
        safe_label = label.replace(' ', '_').replace('[', '').replace(']', '').replace('-', '_')
        fig, axs = plt.subplots(6, 1, figsize=(17, 14), sharex=True)
        axs[0].errorbar(
            dates, flux[:, bi], yerr=err[:, bi],
            fmt='.', color='green', ecolor='green', markersize=1.3,
            elinewidth=0.35, capsize=0, alpha=0.45, errorevery=7,
            label='Daily flux'
        )
        axs[0].plot(dates, flux_smooth[:, bi], color='black', lw=1.2, label='Smoothed flux')
        axs[0].set_ylabel('Flux')
        axs[0].set_title(label, fontsize=13, fontweight='bold')
        axs[0].legend(loc='upper right', fontsize=9)

        for pi, para_label in enumerate(SUN_LABELS):
            ax = axs[pi + 1]
            ax.plot(dates, sun[:, pi], color=SUN_COLORS[pi], lw=0.55, alpha=0.45)
            ax.plot(dates, sun_smooth[:, pi], color=SUN_COLORS[pi], lw=1.2)
            ax.set_ylabel(para_label)

        for ax in axs:
            add_split_lines(ax, dates, train_end, val_end)
            ax.grid(alpha=0.18, lw=0.5)
        axs[-1].set_xlabel('Year')
        fig.suptitle('Electron flux and solar parameters', fontsize=16)
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        fig.savefig(FIG_DIR / f'timeseries_facets_{safe_label}.pdf', bbox_inches='tight')
        plt.close(fig)


def save_best_lag_scatter(dates, flux, err, sun, summary_df,
                          plot_bins, bin_labels, train_end, val_end):
    split_masks = {
        'train': np.arange(len(dates)) < train_end,
        'validation': (np.arange(len(dates)) >= train_end) & (np.arange(len(dates)) < val_end),
        'test': np.arange(len(dates)) >= val_end,
    }

    for bi, label in zip(plot_bins, bin_labels):
        safe_label = label.replace(' ', '_').replace('[', '').replace(']', '').replace('-', '_')
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        axs = axs.reshape(-1)
        for pi, para in enumerate(SUN_SHORT):
            ax = axs[pi]
            row = summary_df[
                (summary_df['bin_index'] == bi) &
                (summary_df['solar_parameter'] == para)
            ].iloc[0]
            lag = int(row['weighted_best_lag_days']) if np.isfinite(row['weighted_best_lag_days']) else 0
            y, x, e = lagged_arrays(flux[:, bi], sun[:, pi], err[:, bi], lag)
            date_lagged = dates[lag:]
            for split_name, mask_all in split_masks.items():
                mask = mask_all[lag:]
                valid = mask & np.isfinite(x) & np.isfinite(y) & np.isfinite(e) & (e >= 0)
                ax.errorbar(
                    x[valid], y[valid], yerr=e[valid],
                    fmt='.', color=SPLIT_COLORS[split_name], ecolor=SPLIT_COLORS[split_name],
                    markersize=2.2, elinewidth=0.35, capsize=0, alpha=0.38,
                    label=split_name if pi == 0 else None,
                )
            ax.set_xlabel(f'{para}(t - {lag} d)')
            ax.set_ylabel('Flux')
            ax.set_title(
                f'{para}: best weighted corr={row["weighted_best_corr"]:.3f}',
                fontsize=11,
            )
            ax.grid(alpha=0.18, lw=0.5)
        axs[5].axis('off')
        handles, labels = axs[0].get_legend_handles_labels()
        if handles:
            axs[5].legend(handles, labels, loc='center', fontsize=11)
        fig.suptitle(f'Best-lag solar parameter scatter: {label}', fontsize=16)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        fig.savefig(FIG_DIR / f'best_lag_scatter_{safe_label}.pdf', bbox_inches='tight')
        plt.close(fig)


def save_zero_lag_scatter(dates, flux, err, sun, corr,
                          plot_bins, bin_labels, train_end, val_end):
    split_masks = {
        'train': np.arange(len(dates)) < train_end,
        'validation': (np.arange(len(dates)) >= train_end) & (np.arange(len(dates)) < val_end),
        'test': np.arange(len(dates)) >= val_end,
    }

    for bi, label in zip(plot_bins, bin_labels):
        safe_label = label.replace(' ', '_').replace('[', '').replace(']', '').replace('-', '_')
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        axs = axs.reshape(-1)
        for pi, para in enumerate(SUN_SHORT):
            ax = axs[pi]
            x = sun[:, pi]
            y = flux[:, bi]
            e = err[:, bi]
            for split_name, mask in split_masks.items():
                valid = mask & np.isfinite(x) & np.isfinite(y) & np.isfinite(e) & (e >= 0)
                ax.errorbar(
                    x[valid], y[valid], yerr=e[valid],
                    fmt='.', color=SPLIT_COLORS[split_name], ecolor=SPLIT_COLORS[split_name],
                    markersize=2.2, elinewidth=0.35, capsize=0, alpha=0.38,
                    label=split_name if pi == 0 else None,
                )
            ax.set_xlabel(f'{para}(t)')
            ax.set_ylabel('Flux')
            ax.set_title(f'{para}: corr(lag=0)={corr[bi, pi, 0]:.3f}', fontsize=11)
            ax.grid(alpha=0.18, lw=0.5)
        axs[5].axis('off')
        handles, labels = axs[0].get_legend_handles_labels()
        if handles:
            axs[5].legend(handles, labels, loc='center', fontsize=11)
        fig.suptitle(f'Zero-lag solar parameter scatter: {label}', fontsize=16)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        fig.savefig(FIG_DIR / f'zero_lag_scatter_{safe_label}.pdf', bbox_inches='tight')
        plt.close(fig)


def main():
    args = parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    flux = np.load(FLUX_PATH)
    err = np.load(ERR_PATH)
    energy_edges = np.load(ENERGY_PATH)
    sun_all = np.load(SUN_PATH)[:, 0:5]

    sun_offset = (FLUX_START - SUN_START).days
    n_days = min(flux.shape[0], err.shape[0], sun_all.shape[0] - sun_offset)
    if n_days <= args.max_lag + args.min_points:
        raise ValueError('Not enough overlapping flux/solar days for the requested lag scan.')
    if n_days < flux.shape[0]:
        print(f'Trim to overlapping days: {n_days} from flux length {flux.shape[0]}')
    flux = flux[:n_days]
    err = err[:n_days]
    sun = sun_all[sun_offset:sun_offset + n_days]
    dates = pd.date_range(FLUX_START, periods=n_days, freq='D')

    split = (args.train_num, args.val_num)
    if args.train_num is None or args.val_num is None:
        parsed = parse_best_split()
        split = parsed if parsed is not None else (0.6, 0.2)
    train_num, val_num = split
    train_end, val_end, test_end = split_info(n_days, train_num, val_num)
    print('flux shape:', flux.shape)
    print('err shape:', err.shape)
    print('sun shape:', sun.shape)
    print('date range:', dates[0].date(), dates[-1].date())
    print('train/val/test:', train_end, val_end - train_end, test_end - val_end)

    energy_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    targets = [1.0, 2.0, 5.0, 10.0]
    plot_bins = [int(np.argmin(np.abs(energy_centers - target))) for target in targets]
    bin_labels = [
        f'{energy_centers[bi]:.2f} GeV [{energy_edges[bi]:.2f}-{energy_edges[bi + 1]:.2f}]'
        for bi in plot_bins
    ]

    flux_smooth = np.column_stack([moving_mean(flux[:, bi], args.smooth_days) for bi in range(flux.shape[1])])
    sun_smooth = np.column_stack([moving_mean(sun[:, pi], args.smooth_days) for pi in range(sun.shape[1])])

    lags = np.arange(0, args.max_lag + 1, args.lag_step, dtype=int)
    corr, wcorr = compute_lag_correlations(flux_smooth, sun_smooth, err, lags, args.min_points)
    summary_df = best_lag_summary(corr, wcorr, lags, plot_bins, bin_labels)
    summary_path = FIG_DIR / 'lag_correlation_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print('summary:', summary_path)

    save_faceted_timeseries(
        dates, flux, err, sun, flux_smooth, sun_smooth,
        plot_bins, bin_labels, train_end, val_end,
    )
    save_zscore_overlay(dates, flux_smooth, sun_smooth, plot_bins, bin_labels, train_end, val_end)
    save_lag_corr(lags, corr, plot_bins, bin_labels, 'unweighted', 'Lag correlation: unweighted')
    save_lag_corr(lags, wcorr, plot_bins, bin_labels, 'weighted', 'Lag correlation: weighted by flux error')
    save_zero_lag_scatter(dates, flux, err, sun, corr, plot_bins, bin_labels, train_end, val_end)
    save_best_lag_scatter(dates, flux, err, sun, summary_df, plot_bins, bin_labels, train_end, val_end)

    print('figures:', FIG_DIR)
    print('Done')


if __name__ == '__main__':
    main()
