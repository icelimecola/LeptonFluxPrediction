#!/bin/python
# -*- coding: utf-8 -*-

from datetime import date, datetime, timedelta
from pathlib import Path
import re

import numpy as np


# =====================================================
# 0. 路径与时间范围设置
# =====================================================
# 太阳参数 rawdata 按更新时间分文件夹保存。
# 当前使用 260701，也就是 2026-07-01 这一版下载/整理后的结果。
RAWDATA_DIR = Path('../rawdata/sunpara/260701')
OUT_DIR = Path('../sun_processed/latest')
FIG_DIR = Path('./Figure/sunpara')

# 观测段输入：
#   OMNI  daily data       -> BE, VSW
#   WSO   27-day tilt      -> TILT
#   WSO   10-day polar     -> SP 的覆盖时间
#   SILSO daily sunspots   -> SSN
OMNI_FILE = RAWDATA_DIR / 'omni_m_daily.txt'
TILT_FILE = RAWDATA_DIR / 'wso_tilt_27d.txt'
POLAR_FILE = RAWDATA_DIR / 'wso_polar_10d.txt'
SSN_FILE = RAWDATA_DIR / 'silso_SN_d_tot_V2.0.csv'

# forecast 输入。这里仍然使用手工整理的月度预测表；
# 脚本只会根据最新观测结束日期，自动裁掉已经被观测覆盖的 forecast 日期。
PREDICT_FILE = RAWDATA_DIR / 'para_solar_predict.txt'

# 最终输出保持训练脚本使用的总时间范围不变。
# 观测段/预测段的分界点由当前观测数据自动决定。
OBS_START = date(2010, 5, 20)
PREDICT_TABLE_START = date(2025, 1, 1)
FINAL_END = date(2031, 12, 1)


# =====================================================
# 1. 日期和插值工具函数
# =====================================================
def parse_colon_date(text):
    return datetime.strptime(text, '%Y:%m:%d').date()


def parse_polar_time(text):
    return datetime.strptime(text.split('_')[0], '%Y:%m:%d').date()


def date_range(start, end):
    n_days = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(n_days)]


def daily_interp(dates, values):
    """把非 daily 序列用线性插值转成 daily 序列。"""
    daily_dates = []
    x_daily = []
    x = np.arange(len(values))
    for i in range(len(x) - 1):
        n_interp = (dates[i + 1] - dates[i]).days
        daily_dates.extend(dates[i] + timedelta(days=j) for j in range(n_interp))
        x_daily.extend(np.linspace(x[i], x[i + 1], num=n_interp, endpoint=False))
    daily_dates.append(dates[-1])
    x_daily.append(x[-1])
    return daily_dates, np.interp(np.array(x_daily), x, values)


def select_by_date(dates, values, wanted_dates, label):
    """按日期选取连续 daily 窗口，避免继续依赖固定行号。"""
    index = {d: i for i, d in enumerate(dates)}
    missing = [d for d in wanted_dates if d not in index]
    if missing:
        raise ValueError(f'{label} missing date range: {missing[0]} to {missing[-1]}')
    return np.array([values[index[d]] for d in wanted_dates])


def parse_number_token(token):
    """读取 WSO 里的 '-26Avgf' 这类 token；'XXXAvgf' 这类缺测值记为 nan。"""
    match = re.search(r'[-+]?\d+(?:\.\d+)?', token)
    if match is None:
        return np.nan
    return float(match.group(0))


# =====================================================
# 2. 观测数据读取函数
# =====================================================
# 每个 loader 都同时返回日期和后面要用的物理量。
# 日期轴保留下来之后，观测时间范围就可以自动更新，不需要手动改固定行号切片。
def load_omni(path):
    """NASA/SPDF OMNI daily data: 第 8 列 = BE，第 9 列 = VSW。"""
    data = np.genfromtxt(path)
    years = data[:, 0].astype(int)
    doy = data[:, 1].astype(int)
    dates = [date(y, 1, 1) + timedelta(days=int(d) - 1) for y, d in zip(years, doy)]
    return dates, data[:, 8], data[:, 9]


def load_tilt(path):
    """WSO HCS tilt table: 使用 L_av 列，并从 Carrington rotation 线性插值到 daily。"""
    dates = []
    tilt_values = []
    with path.open(encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 9 or parts[0] != 'CR':
                continue
            dates.append(parse_colon_date(parts[2]))
            tilt_values.append(float(parts[4]))
    tilt_values = np.array(tilt_values, dtype=float)
    daily_dates, daily_values = daily_interp(dates, tilt_values)
    return daily_dates, np.round(daily_values, 1), dates[-1]


def load_polar(path):
    """WSO polar field table。这里主要用日期判断覆盖范围；SP 本身在后面用 sigmoid 平滑。"""
    dates = []
    values = []
    with path.open(encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.split()
            if not parts or not re.match(r'\d{4}:\d{2}:\d{2}_', parts[0]):
                continue
            dates.append(parse_polar_time(parts[0]))
            values.append(parse_number_token(parts[8]))

    values = np.array(values, dtype=float)
    patch = {
        1342: -1.0, 1343: 1.0, 1593: 1.0, 1594: 1.0,
        1690: 1.0, 1691: 1.0, 1692: 1.0, 1693: 1.0,
        1694: 1.0, 1695: 1.0, 1708: 1.0, 1726: 1.0,
        1727: 1.0, 1728: -1.0, 1729: -1.0, 1741: -1.0,
        1742: -1.0, 1743: -1.0, 1744: -1.0,
    }
    for idx, value in patch.items():
        if idx < len(values):
            values[idx] = value
    return dates, values


def load_ssn(path):
    """SILSO daily total sunspot number: 第 4 列 = SSN。"""
    data = np.genfromtxt(path, delimiter=';')
    years = data[:, 0].astype(int)
    months = data[:, 1].astype(int)
    days = data[:, 2].astype(int)
    dates = [date(y, m, d) for y, m, d in zip(years, months, days)]
    return dates, data[:, 4]


# =====================================================
# 3. 太阳磁极性 SP 的 sigmoid 平滑
# =====================================================
# 沿用旧工作流：用两个 sigmoid function 表示太阳磁极性翻转。
# 翻转中心保持不变，这样物理处理方式仍和旧脚本对齐；
# 这次只让观测段长度随新 rawdata 自动变化。
def sigmoid(x, k=0.03, x0=0):
    return 1 / (1 + np.exp(-k * (x - x0)))


def build_sp_sigmoid(n_days):
    x = np.arange(n_days)
    return -1 + 2 * sigmoid(x, k=0.03, x0=1022) - 2 * sigmoid(x, k=0.03, x0=4872)


# =====================================================
# 4. forecast 预测段
# =====================================================
# 月度预测表从 2025-01 开始。
# 当观测结束日期 obs_end 确定后，所有已经被观测覆盖的 forecast 日期都会被丢掉。
def build_forecast(obs_end):
    sun = np.genfromtxt(PREDICT_FILE, skip_header=1)
    sun = sun[60:]
    years = sun[:, 0].astype(int)
    months = sun[:, 1].astype(int)
    month_dates = [date(y, m, 1) for y, m in zip(years, months)]

    forecast_dates = date_range(PREDICT_TABLE_START, FINAL_END)
    keep_dates = [d for d in forecast_dates if d > obs_end]

    def interp_column(values):
        daily_dates, daily_values = daily_interp(month_dates, values)
        return select_by_date(daily_dates, daily_values, keep_dates, 'forecast')

    ssn_daily = interp_column(sun[:, 2])
    tilt_daily = interp_column(sun[:, 4])
    vsw_daily = interp_column(sun[:, 5])
    be_daily = interp_column(sun[:, 6])
    sp_daily = np.ones(len(keep_dates)) * -1.0

    sun4_daily = np.column_stack([be_daily, vsw_daily, tilt_daily, sp_daily])
    sun5_daily = np.column_stack([be_daily, vsw_daily, tilt_daily, sp_daily, ssn_daily])
    return keep_dates, be_daily, vsw_daily, tilt_daily, sp_daily, ssn_daily, sun4_daily, sun5_daily


# =====================================================
# 5. 输出工具函数
# =====================================================
def save(name, array):
    np.save(OUT_DIR / name, array)


def write_sunpara_html(output_path, obs_dates, pred_dates, sun5_daily, sun5_pred):
    from tool_plotlyhtml import make_trace, write_plotly_panels

    labels = ['B (nT)', 'V_sw (km/s)', 'alpha (degree)', 'A', 'SSN']
    panels = []
    for i, label in enumerate(labels):
        panels.append({
            'title': label,
            'traces': [
                make_trace(
                    'Observation', obs_dates, sun5_daily[:, i],
                    mode='lines', color='green', showlegend=(i == 0),
                    hovertemplate=(
                        f'{label}<br>'
                        'date=%{x|%Y-%m-%d}<br>'
                        'Observation=%{y:.6g}<extra></extra>'
                    ),
                ),
                make_trace(
                    'Theory Prediction', pred_dates, sun5_pred[:, i],
                    mode='lines', color='red', dash='dash', showlegend=(i == 0),
                    hovertemplate=(
                        f'{label}<br>'
                        'date=%{x|%Y-%m-%d}<br>'
                        'Theory Prediction=%{y:.6g}<extra></extra>'
                    ),
                ),
            ],
        })
    write_plotly_panels(
        output_path, 'Daily Solar Activity', panels,
        columns=1, yaxis_title='Value', height=260,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # 6. 读取所有观测源
    # =====================================================
    omni_dates, be_all, vsw_all = load_omni(OMNI_FILE)
    tilt_dates_daily, tilt_all, tilt_last = load_tilt(TILT_FILE)
    polar_dates, _ = load_polar(POLAR_FILE)
    ssn_dates, ssn_all = load_ssn(SSN_FILE)

    # =====================================================
    # 7. 定义更新后的观测时间窗口
    # =====================================================
    # 观测段的最后一天由四个观测源里“最短”的那个决定。
    # 对 260701 这版数据来说，最短的是 WSO tilt，结束于 2025-09-09。
    obs_end = min(tilt_last, polar_dates[-1], ssn_dates[-1], omni_dates[-1], FINAL_END)
    obs_dates = date_range(OBS_START, obs_end)
    obs_len = len(obs_dates)

    print('observation range:', OBS_START, 'to', obs_end, obs_len, 'days')
    print('forecast range   :', obs_end + timedelta(days=1), 'to', FINAL_END)

    # =====================================================
    # 8. 构造 daily 观测数组
    # =====================================================
    # BE、VSW、SSN 本身就是 daily 数据。
    # TILT 从 Carrington rotation 表线性插值到 daily。
    # SP 使用旧代码中的 sigmoid-smoothed polarity 曲线。
    be_daily = select_by_date(omni_dates, be_all, obs_dates, 'BE')
    vsw_daily = select_by_date(omni_dates, vsw_all, obs_dates, 'VSW')
    tilt_daily = select_by_date(tilt_dates_daily, tilt_all, obs_dates, 'TILT')
    sp_daily_sigmoid = build_sp_sigmoid(obs_len)
    ssn_daily = select_by_date(ssn_dates, ssn_all, obs_dates, 'SSN')

    sun4_daily = np.column_stack([be_daily, vsw_daily, tilt_daily, sp_daily_sigmoid])
    sun5_daily = np.column_stack([be_daily, vsw_daily, tilt_daily, sp_daily_sigmoid, ssn_daily])

    # =====================================================
    # 9. 构造 daily forecast 数组
    # =====================================================
    # forecast 从 obs_end + 1 天开始，因此最终拼接时观测段和预测段不会重叠。
    pred_dates, be_pred, vsw_pred, tilt_pred, sp_pred, ssn_pred, sun4_pred, sun5_pred = build_forecast(obs_end)

    # =====================================================
    # 10. 拼接观测段和 forecast 段
    # =====================================================
    # 总 shape 仍保持为 (7866, 5)，这样后续 train/draw 脚本读取
    # sun5_daily_all_latest.npy 的接口不需要变化。
    sun4_all = np.concatenate([sun4_daily, sun4_pred], axis=0)
    sun5_all = np.concatenate([sun5_daily, sun5_pred], axis=0)

    # =====================================================
    # 11. 保存所有输出数组
    # =====================================================
    save('be_daily_latest.npy', be_daily)
    save('vsw_daily_latest.npy', vsw_daily)
    save('tilt_daily_latest.npy', tilt_daily)
    save('sp_daily_sigmoid_latest.npy', sp_daily_sigmoid)
    save('ssn_daily_latest.npy', ssn_daily)
    save('sun4_daily_latest.npy', sun4_daily)
    save('sun5_daily_latest.npy', sun5_daily)

    save('be_daily_predict_latest.npy', be_pred)
    save('vsw_daily_predict_latest.npy', vsw_pred)
    save('tilt_daily_predict_latest.npy', tilt_pred)
    save('sp_daily_predict_latest.npy', sp_pred)
    save('ssn_daily_predict_latest.npy', ssn_pred)
    save('sun4_daily_predict_latest.npy', sun4_pred)
    save('sun5_daily_predict_latest.npy', sun5_pred)

    save('sun4_daily_all_latest.npy', sun4_all)
    save('sun5_daily_all_latest.npy', sun5_all)

    print('sun5 observation:', sun5_daily.shape)
    print('sun5 forecast   :', sun5_pred.shape)
    print('sun5 all        :', sun5_all.shape)
    write_sunpara_html(FIG_DIR / 'sun5_latest_24.html', obs_dates, pred_dates, sun5_daily, sun5_pred)

    # =====================================================
    # 12. 可选的检查图
    # =====================================================
    # 画图不是必需步骤：即使运行环境没有 matplotlib，也应该先正常生成 .npy 数据。
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available; skip sun-parameter check figure.')
        return

    all_dates = obs_dates + pred_dates
    fig, axs = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
    fig.suptitle('Daily solar activity', fontsize=16)
    labels = ['B (nT)', 'V_sw (km/s)', 'alpha (degree)', 'A', 'SSN']
    for i, label in enumerate(labels):
        axs[i].plot(obs_dates, sun5_daily[:, i], 'g-', label='Observation')
        axs[i].plot(pred_dates, sun5_pred[:, i], 'r--', label='Theory Prediction')
        axs[i].set_ylabel(label)
    axs[0].legend(loc='upper right')
    axs[-1].set_xlabel('Year')
    axs[-1].set_xlim(all_dates[0], all_dates[-1])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(FIG_DIR / 'sun5_latest_24.pdf')
    plt.close()


if __name__ == '__main__':
    main()
