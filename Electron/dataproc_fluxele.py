#!/bin/python

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import uproot


def fill_gaps_with_neighbor_sum(series):
    """
    填补连续 NaN 块：
    - 中间块：用前后值之和填充
    - 开头/结尾块：用存在一侧的值的 1 倍填充
    - 全 NaN: 填充 0

    Parameters:
        series (pd.Series): 输入序列

    Returns:
        pd.Series: 填充后的序列，无 NaN
    """
    if not isinstance(series, pd.Series):
        raise TypeError("输入必须是 pandas Series")

    result = series.copy()
    values = result.values
    n = len(values)

    i = 0
    while i < n:
        if not pd.isna(values[i]):
            i += 1
            continue

        # 找出连续 NaN 块
        start = i
        while i < n and pd.isna(values[i]):
            i += 1
        end = i - 1

        prev_idx = start - 1
        next_idx = end + 1

        prev_val = None
        next_val = None

        if prev_idx >= 0 and not pd.isna(values[prev_idx]):
            prev_val = values[prev_idx]
        if next_idx < n and not pd.isna(values[next_idx]):
            next_val = values[next_idx]

        if prev_val is not None and next_val is not None:
            fill_value = prev_val + next_val
        elif prev_val is not None:
            fill_value = 1 * prev_val
        elif next_val is not None:
            fill_value = 1 * next_val
        else:
            fill_value = 0.0

        result.iloc[start:end+1] = fill_value

    return result


# =====================================================
# 1. 读取 ROOT 文件
# =====================================================
f = uproot.open("../rawdata/flux/eleflux.root")
h2 = f["hflux_2d;1"]

flux_2d = h2.values()   # (6000, 42)
err_2d  = h2.errors()   # (6000, 42)

print("Raw flux shape:", flux_2d.shape)
print("Raw error shape:", err_2d.shape)

# =====================================================
# 2. 获取坐标信息
# =====================================================
# 时间轴 (Unix 时间戳)
x_low   = h2.axis(0).member('fXmin')
x_high  = h2.axis(0).member('fXmax')
x_nbins = h2.axis(0).member('fNbins')
bin_width = (x_high - x_low) / x_nbins  # 86400s = 1 day

# 能量轴
y_low   = h2.axis(1).member('fXmin')
y_high  = h2.axis(1).member('fXmax')
y_nbins = h2.axis(1).member('fNbins')
# 能量轴 (变宽 bin)
y_edges = np.array(h2.axis(1).member('fXbins'))  # 43 edges, 42 bins

# =====================================================
# 3. 能量筛选: 只取 >= 1 GeV 的 bin (跳过 bin 0)
# =====================================================
# bin 0: [0.8, 1.0] GeV, bin 1 起 >= 1.0 GeV
keep_bins = np.arange(1, y_nbins)  # bins 1..41
flux_2d = flux_2d[:, keep_bins]
err_2d  = err_2d[:, keep_bins]
y_edges_kept = y_edges[keep_bins[0]:]  # edges for kept bins

n_days, n_bins = flux_2d.shape
print(f"After E cut (>={y_edges_kept[0]:.1f} GeV): ({n_days}, {n_bins})")

# =====================================================
# 4. 生成日期索引
# =====================================================
# bin 中心是日期，从第一个 bin 中心到最后一个 bin 中心
from datetime import datetime, timezone
first_center = x_low + bin_width / 2

def timestamp_to_date(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

start_date = timestamp_to_date(first_center)
# 最后一天
last_center = x_high - bin_width / 2
end_date = timestamp_to_date(last_center)

print(f"Date range: {start_date} ~ {end_date}")

idx = pd.date_range(start=start_date, end=end_date, freq='D')
print(f"idx length: {len(idx)}, data rows: {n_days}")

# 截断到数据长度 (以防日期索引比数据长 1)
idx = idx[:n_days]

# =====================================================
# 5. 剔除头尾全零段
# =====================================================
# 找到第一个和最后一个有有效值的日期
has_data = np.sum(flux_2d, axis=1) > 0
valid_rows = np.where(has_data)[0]
first_valid = valid_rows[0]
last_valid = valid_rows[-1]

print(f"Head zero block: {first_valid} days, Tail zero block: {n_days - last_valid - 1} days")

# 截取有效范围
flux_2d = flux_2d[first_valid:last_valid+1, :]
err_2d  = err_2d[first_valid:last_valid+1, :]
idx     = idx[first_valid:last_valid+1]
n_days  = flux_2d.shape[0]

print(f"After trim: {n_days} days, {idx[0]} ~ {idx[-1]}")

# =====================================================
# 6. 剩余 0 值 → NaN (中间缺失)
# =====================================================
# 真实电子流强不可能为 0 → 所有 0 都是缺失
zero_mask = flux_2d == 0
flux_2d[zero_mask] = np.nan
err_2d[zero_mask] = np.nan
print(f"Zero values → NaN: {int(np.sum(zero_mask))} / {flux_2d.size} ({int(np.sum(zero_mask))/flux_2d.size*100:.1f}%)")

# =====================================================
# 7. 逐 bin 插值 flux & 填充误差
# =====================================================
flux_filled = np.zeros((n_days, n_bins))
err_filled  = np.zeros((n_days, n_bins))

for i in range(n_bins):
    # flux: 线性插值 + 边界用最近邻填充 (头尾大块 NaN)
    s_flux = pd.Series(flux_2d[:, i], index=idx)
    s_flux = s_flux.interpolate(limit_direction='both')
    s_flux = s_flux.fillna(method='ffill').fillna(method='bfill')
    flux_filled[:, i] = s_flux.values

    # error: 用 neighbor_sum 填充 (保守估计)
    s_err = pd.Series(err_2d[:, i], index=idx)
    err_filled[:, i] = fill_gaps_with_neighbor_sum(s_err).values

print("Flux NaN after interpolation:", int(np.sum(np.isnan(flux_filled))))
print("Error NaN after fill:        ", int(np.sum(np.isnan(err_filled))))

# =====================================================
# 8. 计算误差
# =====================================================
abs_error  = err_filled
rela_error = err_filled / flux_filled  # 相对误差
rela_error[np.isinf(rela_error)] = np.nan

# =====================================================
# 9. 30 天滑动平均
# =====================================================
window_size = 30
flux_ave = np.zeros((n_days, n_bins))
for i in range(n_bins):
    for j in range(n_days):
        flux_ave[j, i] = np.mean(flux_filled[j:j+window_size, i])

# =====================================================
# 10. 保存 npy
# =====================================================
import os
os.makedirs("Data/flux", exist_ok=True)
os.makedirs("Figure/flux", exist_ok=True)

np.save("Data/flux/electron_flux_allbin.npy", flux_filled)
np.save("Data/flux/electron_flux_abs_error_allbin.npy", abs_error)
np.save("Data/flux/electron_flux_rela_error_allbin.npy", rela_error)
np.save("Data/flux/electron_flux_ave_allbin.npy", flux_ave)
np.save("Data/flux/electron_energy_edges.npy", y_edges_kept)

print("\nSaved:")
print(f"  Data/flux/electron_flux_allbin.npy            {flux_filled.shape}")
print(f"  Data/flux/electron_flux_abs_error_allbin.npy  {abs_error.shape}")
print(f"  Data/flux/electron_flux_rela_error_allbin.npy {rela_error.shape}")
print(f"  Data/flux/electron_flux_ave_allbin.npy        {flux_ave.shape}")
print(f"  Data/flux/electron_energy_edges.npy           {y_edges_kept.shape}")

# =====================================================
# 11. 画图 (4 代表能档: ~1, 2, 5, 10 GeV, 2×2)
# =====================================================
energy_centers = 0.5 * (y_edges_kept[:-1] + y_edges_kept[1:])
targets = [1.0, 2.0, 5.0, 10.0]
plot_bins = [int(np.argmin(np.abs(energy_centers - t))) for t in targets]

plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
for j, bi in enumerate(plot_bins):
    ax = axs[j // 2, j % 2]
    ax.plot(idx, flux_filled[:, bi], lw=0.5)
    ax.set_title(f'{energy_centers[bi]:.2f} GeV  [{y_edges_kept[bi]:.2f}–{y_edges_kept[bi+1]:.2f}] GeV', fontsize=11)
    ax.set_ylabel('Flux')

plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.08, wspace=0.08, hspace=0.15)
fig.supxlabel('Year', y=0.03, fontsize=13)
fig.supylabel('Electron Flux  [m$^{-2}$ s$^{-1}$ sr$^{-1}$ (GeV/n)$^{-1}$]', x=0.03, fontsize=13)
plt.savefig('Figure/flux/electron_flux_overview.pdf', bbox_inches='tight')
plt.close()

print("\nPlots saved to Figure/flux/")
print("Done ✅")
