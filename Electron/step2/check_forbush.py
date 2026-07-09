#!/bin/python
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import uproot


# =====================================================
# 1. 读取 electron flux ROOT 文件
# =====================================================
# ROOT 文件沿用 dataproc_fluxele.py 的输入：
#   hflux_2d axis 0 -> time bin，单位是 Unix timestamp
#   hflux_2d axis 1 -> energy bin，单位是 GeV
# 这里同时读取 flux values 和 bin errors，后面直接用于 errorbar。
def read_flux_from_root(root_file):
    f = uproot.open(root_file)
    h2 = f["hflux_2d;1"]

    flux_2d = h2.values()
    err_2d = h2.errors()

    x_axis = h2.axis(0)
    x_low = x_axis.member("fXmin")
    x_high = x_axis.member("fXmax")
    x_nbins = x_axis.member("fNbins")
    bin_width = (x_high - x_low) / x_nbins

    first_center = x_low + bin_width / 2
    last_center = x_high - bin_width / 2
    start_date = datetime.fromtimestamp(first_center, tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None
    )
    end_date = datetime.fromtimestamp(last_center, tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None
    )
    idx = np.array([start_date + timedelta(days=i) for i in range(flux_2d.shape[0])])

    y_edges = np.array(h2.axis(1).member("fXbins"))

    # 与 dataproc_fluxele.py 保持一致：跳过 [0.8, 1.0] GeV，只保留 >= 1 GeV。
    keep_bins = np.arange(1, h2.axis(1).member("fNbins"))
    flux_2d = flux_2d[:, keep_bins]
    err_2d = err_2d[:, keep_bins]
    y_edges = y_edges[keep_bins[0] :]

    # electron flux 不应为 0。ROOT 里的 0 按 missing data 处理，画图时不连假点。
    zero_mask = flux_2d == 0
    flux_2d = flux_2d.astype(float)
    err_2d = err_2d.astype(float)
    flux_2d[zero_mask] = np.nan
    err_2d[zero_mask] = np.nan

    return idx, flux_2d, err_2d, y_edges


# =====================================================
# 2. 构造 Forbush decrease 检查窗口
# =====================================================
# 输入中心日期 center_date，默认取前后各 6 天，总共 13 天。
# 这样图上可以直接看 center 附近是否快速 decrease 后又 recovery。
def make_window(center_date, idx, days=13):
    center = datetime.strptime(center_date, "%Y-%m-%d")
    half = days // 2
    start = center - timedelta(days=half)
    end = center + timedelta(days=half)

    if start < idx[0] or end > idx[-1]:
        raise ValueError(
            "Requested window is outside data range: "
            f"{start.date()} ~ {end.date()}, data range {idx[0].date()} ~ {idx[-1].date()}"
        )

    mask = (idx >= start) & (idx <= end)
    return center, mask


# =====================================================
# 3. 画四个代表 energy bin 的 flux + error
# =====================================================
# targets 默认是 1, 2, 5, 10 GeV。
# 实际使用 ROOT 里最接近这些目标能量的 energy bin。
# 每个 panel 用点表示 daily flux，errorbar 表示 bin error，并用折线连接。
def draw_forbush(center_date, root_file, outdir, targets):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from tool_plotlyhtml import make_trace, write_plotly_panels

    idx, flux_2d, err_2d, energy_edges = read_flux_from_root(root_file)
    center, mask = make_window(center_date, idx, days=13)

    energy_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    plot_bins = [int(np.argmin(np.abs(energy_centers - t))) for t in targets]

    # =====================================================
    # 4. matplotlib 画图格式
    # =====================================================
    # 布局仿照 lstm_draw.py 最后的 2x2 panel。
    # 中心日期用竖虚线标出，方便检查 Forbush decrease 的最低点和恢复趋势。
    x = idx[mask]
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10

    fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    for j, bi in enumerate(plot_bins):
        ax = axs[j // 2, j % 2]
        y = flux_2d[mask, bi]
        yerr = err_2d[mask, bi]

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            markersize=4,
            lw=0.9,
            elinewidth=0.8,
            capsize=2,
            color="tab:blue",
            ecolor="tab:blue",
            alpha=0.9,
        )
        ax.axvline(center, color="0.3", ls=":", lw=1.0, alpha=0.8)
        ax.text(
            0.02,
            0.88,
            "%.2f GeV  [%.2f-%.2f] GeV" % (energy_centers[bi], energy_edges[bi], energy_edges[bi + 1]),
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel("Flux")
        ax.grid(True, color="0.9", lw=0.6)

    # x-axis 只在底部两个 panel 显示 daily tick，并保留年份。
    for ax in axs[-1, :]:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
        ax.tick_params(axis="x", rotation=45)

    # PDF 输出统一放在 Figure/check_fd，方便和其他 Figure 分类保存。
    outdir.mkdir(parents=True, exist_ok=True)
    date_tag = center.strftime("%Y%m%d")
    pdf_file = outdir / f"electron_forbush_decrease_{date_tag}.pdf"
    html_file = outdir / f"electron_forbush_decrease_{date_tag}.html"

    plt.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.10, wspace=0.08, hspace=0.12)
    fig.suptitle(f"Electron Flux around {center.strftime('%Y-%m-%d')} (13 days)", fontsize=15)
    fig.supxlabel("Date", y=0.03, fontsize=14)
    fig.supylabel("Electron Flux  [m$^{-2}$ s$^{-1}$ sr$^{-1}$ (GeV/n)$^{-1}$]", x=0.03, fontsize=14)
    plt.savefig(pdf_file, bbox_inches="tight")
    plt.close()

    panels = []
    for j, bi in enumerate(plot_bins):
        title = "%.2f GeV  [%.2f-%.2f] GeV" % (energy_centers[bi], energy_edges[bi], energy_edges[bi + 1])
        y = flux_2d[mask, bi]
        yerr = err_2d[mask, bi]
        panels.append({
            "title": title,
            "traces": [
                make_trace(
                    "flux", x, y, mode="lines+markers", color="#1f77b4",
                    width=0.9, marker_size=5,
                    error_y=yerr, customdata=yerr,
                    hovertemplate=(
                        f"{title}<br>"
                        "date=%{x|%Y-%m-%d}<br>"
                        "flux=%{y:.6g}<br>"
                        "error=%{customdata:.6g}<extra></extra>"
                    ),
                    showlegend=(j == 0),
                )
            ],
            "vlines": [center],
        })
    write_plotly_panels(
        html_file,
        f"Electron Flux around {center.strftime('%Y-%m-%d')} (13 days)",
        panels,
        columns=2,
        yaxis_title="Electron Flux [m^-2 s^-1 sr^-1 (GeV/n)^-1]",
    )

    print("center date =", center.strftime("%Y-%m-%d"))
    print("window      =", x[0].strftime("%Y-%m-%d"), "~", x[-1].strftime("%Y-%m-%d"))
    print("root file   =", root_file)
    print("saved       =", pdf_file)
    print("saved       =", html_file)


# =====================================================
# 5. 命令行参数入口
# =====================================================
# 最常用方式：
#   python3 check_forbush_decrease.py 2017-09-08
# 可选参数：
#   --root-file 指定其他 ROOT 文件
#   --outdir    指定 Figure/check_fd 输出目录
#   --targets   指定四个代表 energy target，例如 1,2,5,10
def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Draw 13-day electron flux with errors around a given Forbush decrease date."
    )
    parser.add_argument("date", help="Center date, e.g. 2017-09-08")
    parser.add_argument(
        "--root-file",
        default=here.parent.parent / "rawdata" / "flux" / "eleflux.root",
        type=Path,
        help="Input ROOT file. Default: ../../rawdata/flux/eleflux.root",
    )
    parser.add_argument(
        "--outdir",
        default=here / "Figure" / "check_fd",
        type=Path,
        help="Output figure directory. Default: ./Figure/check_fd under this script directory",
    )
    parser.add_argument(
        "--targets",
        default="1,2,5,10",
        help="Comma-separated representative energy targets in GeV. Default: 1,2,5,10",
    )
    args = parser.parse_args()

    targets = [float(v) for v in args.targets.split(",")]
    if len(targets) != 4:
        raise ValueError("--targets should contain exactly 4 energies, e.g. 1,2,5,10")

    draw_forbush(args.date, args.root_file, args.outdir, targets)


if __name__ == "__main__":
    main()
