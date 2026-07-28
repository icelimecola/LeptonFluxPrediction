#!/bin/python
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import re


import os
os.makedirs('./Data/lstmdraww', exist_ok=True)
os.makedirs('./Figure/lstmdraww', exist_ok=True)

FLUX_PATH = 'Data/flux/electron_flux_allbin.npy'
ERROR_PATH = 'Data/flux/electron_flux_abs_error_allbin.npy'


def load_energy_edges():
    path = Path('Data/flux/electron_energy_edges.npy')
    if path.exists():
        print('electron energy edges path : ', path)
        return np.load(path)
    raise FileNotFoundError('No electron_energy_edges.npy found in step2.2/Data/flux')

# =====================================================
# 0. 模型选择与 split 识别
# =====================================================
model_list = [
    # 'sunOnlyWeighted_0-5000epoch_0.0001learningRate_64neurons_0.002l2_0.08dropout_64batchSize_0.6train_0.2val_0553-0.00321.keras',
]

model_dir = Path('./Data/modelw')
best_model_file = model_dir / 'best_model.txt'

if not model_list and best_model_file.exists():
    best_model = best_model_file.read_text(encoding='utf-8').strip()
    if best_model:
        model_list = [best_model]

if not model_list:
    raise FileNotFoundError(
        'No weighted model selected. Fill model_list in lstm_draw_w.py or run '
        'lstm_bestmodel_w.py to create Data/modelw/best_model.txt'
    )


MODEL_SPLIT_RE = re.compile(
    r'_(?P<train_num>[-+0-9.eE]+)train_(?P<val_num>[-+0-9.eE]+)val_\d+-[-+0-9.eE]+\.keras$'
)


def parse_model_split(model_name):
    match = MODEL_SPLIT_RE.search(model_name)
    if match is None:
        return 0.6, 0.2
    return float(match.group('train_num')), float(match.group('val_num'))


model_splits = {parse_model_split(m) for m in model_list}
if len(model_splits) != 1:
    raise ValueError('All models in one draw run should use the same train/val split.')
train_num, val_num = next(iter(model_splits))

# =====================================================
# 1. 载入太阳和流强数据
# =====================================================
sun_daily      = np.load('../../sun_processed/latest/sun5_daily_all_latest.npy')
electron_daily = np.load(FLUX_PATH)
electron_error_daily = np.load(ERROR_PATH)

print('sun daily all latest : ', sun_daily.shape)
print('electron flux daily : ',  electron_daily.shape)
print('electron error daily : ',  electron_error_daily.shape)
print('electron flux path : ', FLUX_PATH)
print('electron error path : ', ERROR_PATH)

# 电子前补 365 天零
pad_days  = 365
electron_daily = np.concatenate([np.zeros([pad_days, electron_daily.shape[1]]), electron_daily])
electron_error_daily = np.concatenate([np.zeros([pad_days, electron_error_daily.shape[1]]), electron_error_daily])

# 自动计算 sun offset
flux_start = pd.Timestamp('2011-06-11') - pd.Timedelta(days=pad_days)
sun_start  = pd.Timestamp('2010-05-20')
sun_offset = (flux_start - sun_start).days

number = electron_daily.shape[0]
bins = electron_daily.shape[1]

look_back  = 365
n_features = 5
print('number = ', number, 'bins = ', bins, 'sun_offset =', sun_offset)
print('train_num =', train_num, 'val_num =', val_num)

future_end = sun_daily.shape[0] - sun_offset
X_all_raw = sun_daily[sun_offset:sun_offset+future_end, 0:5]
print('X_all_raw = ', X_all_raw.shape)

# =====================================================
# 2. 划分训练、验证和测试集
# =====================================================
train_end = int(number * train_num)
val_end   = int(number * (train_num + val_num))
test_end  = number
print('train_end, val_end, test_end, future_end:', train_end, val_end, test_end, future_end)

X_train = X_all_raw[0:train_end, :]
y_train = electron_daily[0:train_end, :]
err_train_origin = electron_error_daily[0:train_end, :]
X_val   = X_all_raw[train_end-look_back:val_end, :]
y_val   = electron_daily[train_end-look_back:val_end, :]
err_val_origin = electron_error_daily[train_end-look_back:val_end, :]
X_test  = X_all_raw[val_end-look_back:test_end, :]
y_test  = electron_daily[val_end-look_back:test_end, :]
err_test_origin = electron_error_daily[val_end-look_back:test_end, :]
X_future_raw = X_all_raw[number-look_back:future_end, :]

# =====================================================
# 3. 随机种子
# =====================================================
seed = 42
import random
random.seed(seed)
np.random.seed(seed)

# =====================================================
# 4. 归一化
# =====================================================
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
X_future = scaler.transform(X_future_raw)

scaler_flux = MinMaxScaler()
y_train = scaler_flux.fit_transform(y_train)
y_val   = scaler_flux.transform(y_val)
y_test  = scaler_flux.transform(y_test)

print(np.isnan(X_train).any(), np.isinf(X_train).any())
print(np.isnan(y_train).any(), np.isinf(y_train).any())
print(np.isnan(X_val).any(),   np.isinf(X_val).any())
print(np.isnan(y_val).any(),   np.isinf(y_val).any())

# =====================================================
# 6. 序列化函数
# =====================================================
def make_sequence(X, y, look_back):
    X_seq, y_seq = [], []
    for i in range(look_back, len(X)):
        X_seq.append(X[i - look_back:i, :])
        y_seq.append(y[i, :])
    return np.array(X_seq), np.array(y_seq)


def safe_relative_error(pred, true):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    out = np.full_like(pred, np.nan, dtype=float)
    valid = np.isfinite(pred) & np.isfinite(true) & (true > 0)
    out[valid] = pred[valid] / true[valid] - 1
    return out


def safe_mean_abs_relative_error(pred, true):
    err = safe_relative_error(pred, true)
    return np.nanmean(np.abs(err))

X_train_seq, y_train_seq = make_sequence(X_train, y_train, look_back)
X_val_seq,   y_val_seq   = make_sequence(X_val,   y_val,   look_back)
X_test_seq,  y_test_seq  = make_sequence(X_test,  y_test,  look_back)
X_future_seq, _ = make_sequence(X_future, np.zeros((len(X_future), bins)), look_back)
print(X_train_seq.shape, y_train_seq.shape,
      X_val_seq.shape,   y_val_seq.shape,
      X_test_seq.shape,  y_test_seq.shape,
      X_future_seq.shape)


def write_prediction_html(output_path, idx, plot_bins, bin_labels,
                          train_true, val_true, test_true,
                          train_err, val_err, test_err,
                          train_pred, val_pred, test_pred, future_pred):
    from tool_plotlyhtml import make_trace, write_plotly_panels

    panels = []
    for j, bi in enumerate(plot_bins):
        title = bin_labels[j]

        obs_traces = [
            ('observation', idx[look_back:train_end], train_true[look_back:, bi], train_err[look_back:, bi]),
            ('observation', idx[train_end:val_end], val_true[:, bi], val_err[look_back:, bi]),
            ('observation', idx[val_end:test_end], test_true[:, bi], test_err[look_back:, bi]),
        ]
        traces = []
        for name, x, y, err in obs_traces:
            traces.append(
                make_trace(
                    name, x, y, mode='markers', color='green', showlegend=(j == 0),
                    error_y=err, customdata=err,
                    hovertemplate=(
                        f'{title}<br>'
                        'date=%{x|%Y-%m-%d}<br>'
                        'observation=%{y:.6g}<br>'
                        'error=%{customdata:.6g}<extra></extra>'
                    ),
                )
            )

        pred_traces = [
            ('training prediction', idx[look_back:train_end], train_pred[:, bi], 'blue'),
            ('validation prediction', idx[train_end:val_end], val_pred[:, bi], 'goldenrod'),
            ('test prediction', idx[val_end:test_end], test_pred[:, bi], 'magenta'),
            ('future prediction', idx[test_end:future_end], future_pred[:, bi], 'red'),
        ]
        for name, x, y, color in pred_traces:
            traces.append(
                make_trace(
                    name, x, y, mode='lines', color=color, showlegend=(j == 0),
                    hovertemplate=(
                        f'{title}<br>'
                        'date=%{x|%Y-%m-%d}<br>'
                        f'{name}=%{{y:.6g}}<extra></extra>'
                    ),
                )
            )
        panels.append({
            'title': title,
            'traces': traces,
            'vlines': [idx[split] for split in [look_back, train_end, val_end, test_end]],
        })
    write_plotly_panels(
        output_path, 'Sun-Only Weighted Electron Flux Prediction', panels,
        columns=2, yaxis_title='Electron Flux',
    )


def write_error_html(output_path, idx, plot_bins, bin_labels, train_error, val_error, test_error):
    from tool_plotlyhtml import make_trace, write_plotly_panels

    panels = []
    for j, bi in enumerate(plot_bins):
        title = bin_labels[j]
        raw_traces = [
            ('training error', idx[look_back:train_end], train_error[:, bi], 'blue'),
            ('validation error', idx[train_end:val_end], val_error[:, bi], 'goldenrod'),
            ('test error', idx[val_end:test_end], test_error[:, bi], 'magenta'),
        ]
        traces = []
        for name, x, y, color in raw_traces:
            traces.append(
                make_trace(
                    name, x, y, mode='lines', color=color, showlegend=(j == 0),
                    hovertemplate=(
                        f'{title}<br>'
                        'date=%{x|%Y-%m-%d}<br>'
                        f'{name}=%{{y:.6g}}<extra></extra>'
                    ),
                )
            )
        panels.append({
            'title': title,
            'traces': traces,
            'vlines': [idx[split] for split in [train_end, val_end]],
            'hlines': [0],
        })
    write_plotly_panels(
        output_path, 'Sun-Only Weighted Electron Flux Relative Error', panels,
        columns=2, yaxis_title='Relative Error',
    )

from tensorflow.keras.models import load_model

# =====================================================
# 7. 逐模型预测 & 评估
# =====================================================
for m in model_list:
    model = load_model(str(model_dir / m), compile=False)
    if model.input_shape[-1] != n_features:
        raise ValueError(
            f'Model {m} expects {model.input_shape[-1]} features, '
            f'but current sun-only weighted workflow builds {n_features}.'
        )

    # --- 训练集预测 ---
    train_pred_mean   = model.predict(X_train_seq)
    train_pred_origin = scaler_flux.inverse_transform(train_pred_mean)
    train_true_origin = scaler_flux.inverse_transform(y_train[:, :])
    train_rme = safe_mean_abs_relative_error(train_pred_origin, train_true_origin[look_back:])

    # --- 验证集预测 ---
    val_pred_mean   = model.predict(X_val_seq)
    val_pred_origin = scaler_flux.inverse_transform(val_pred_mean)
    val_true_origin = scaler_flux.inverse_transform(y_val[look_back:, :])
    val_rme = safe_mean_abs_relative_error(val_pred_origin, val_true_origin)

    # --- 测试集预测 ---
    test_pred_mean   = model.predict(X_test_seq)
    test_pred_origin = scaler_flux.inverse_transform(test_pred_mean)
    test_true_origin = scaler_flux.inverse_transform(y_test[look_back:, :])
    test_rme = safe_mean_abs_relative_error(test_pred_origin, test_true_origin)

    print('mean relative error : training', train_rme, ', validation', val_rme, ', test', test_rme)

    # --- 计算各段相对误差 ---
    train_error = safe_relative_error(train_pred_origin, train_true_origin[look_back:])
    val_error   = safe_relative_error(val_pred_origin, val_true_origin)
    test_error  = safe_relative_error(test_pred_origin, test_true_origin)
    print('train error shape =', train_error.shape)

    np.save('./Data/lstmdraww/train_error_allbin_' + m + '.npy', train_error)
    np.save('./Data/lstmdraww/val_error_allbin_'   + m + '.npy', val_error)
    np.save('./Data/lstmdraww/test_error_allbin_'  + m + '.npy', test_error)

    # --- 未来预测：只依赖未来太阳参数，不使用 flux/err 自回归输入 ---
    future_pred_mean = model.predict(X_future_seq)
    future_pred_origin = scaler_flux.inverse_transform(future_pred_mean)

    # =====================================================
    # 9. 画图 — 4 个代表能档: ~1, 2, 5, 10 GeV, 2×2 布局
    # =====================================================
    # 加载真实能量边界
    energy_edges = load_energy_edges()  # (42,) for 41 bins
    energy_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])

    # 选择最接近 1, 2, 5, 10 GeV 的 bin
    targets = [1.0, 2.0, 5.0, 10.0]
    plot_bins = [int(np.argmin(np.abs(energy_centers - t))) for t in targets]

    bin_labels = []
    for bi in plot_bins:
        bin_labels.append('%.2f GeV  [%.2f–%.2f]' % (
            energy_centers[bi], energy_edges[bi], energy_edges[bi+1]))

    idx = pd.date_range(start="2010-06-11", end="2031-12-01")

    plt.rcParams['axes.labelsize']  = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    # —— 预测 vs 真实, 2×2 ——
    fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    for j, bi in enumerate(plot_bins):
        ax = axs[j // 2, j % 2]
        ax.errorbar(idx[look_back:train_end], train_true_origin[look_back:, bi],
                    yerr=err_train_origin[look_back:, bi], fmt='.', color='g',
                    ecolor='g', markersize=1.8, elinewidth=0.45, capsize=0,
                    alpha=0.65, zorder=1)
        ax.errorbar(idx[train_end:val_end], val_true_origin[:, bi],
                    yerr=err_val_origin[look_back:, bi], fmt='.', color='g',
                    ecolor='g', markersize=1.8, elinewidth=0.45, capsize=0,
                    alpha=0.65, zorder=1)
        ax.errorbar(idx[val_end:test_end], test_true_origin[:, bi],
                    yerr=err_test_origin[look_back:, bi], fmt='.', color='g',
                    ecolor='g', markersize=1.8, elinewidth=0.45, capsize=0,
                    alpha=0.65, zorder=1)
        ax.plot(idx[look_back:train_end],  train_pred_origin[:, bi], 'b-', lw=0.5)
        ax.plot(idx[train_end:val_end],      val_pred_origin[:, bi], 'y-', lw=0.5)
        ax.plot(idx[val_end:test_end],      test_pred_origin[:, bi], 'm-', lw=0.5)
        ax.plot(idx[test_end:future_end], future_pred_origin[:, bi], 'r-', lw=0.5)
        ax.text(0.02, 0.88, bin_labels[j], transform=ax.transAxes, fontsize=11, fontweight='bold')

        split_style = dict(color='0.3', ls=':', lw=1.0, alpha=0.7, zorder=0)
        ax.axvline(idx[look_back], **split_style)
        ax.axvline(idx[train_end],   **split_style)
        ax.axvline(idx[val_end],     **split_style)
        ax.axvline(idx[test_end],    **split_style)

        # 底部标注只放一次
        if j >= 2:
            trans = ax.get_xaxis_transform()
            if j == 2:
                pos_train = int((train_end + look_back) / 2)
                pos_val   = int((train_end + val_end) / 2)
                pos_test  = int((val_end + test_end) / 2)
            ax.text(idx[pos_train], 0.03, 'training',   transform=trans, ha='center', va='bottom', fontsize=10)
            ax.text(idx[pos_val],   0.03, 'validation', transform=trans, ha='center', va='bottom', fontsize=10)
            ax.text(idx[pos_test],  0.03, 'test',       transform=trans, ha='center', va='bottom', fontsize=10)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.96, bottom=0.08, wspace=0.06, hspace=0.10)
    fig.supylabel('Electron Flux  [m$^{-2}$ s$^{-1}$ sr$^{-1}$ (GeV/n)$^{-1}$]', x=0.03, fontsize=14)
    fig.supxlabel('Year', y=0.03, fontsize=14)
    plt.savefig('./Figure/lstmdraww/electron_prediction_' + m + '.pdf', bbox_inches='tight')
    plt.close()
    write_prediction_html(
        './Figure/lstmdraww/electron_prediction_' + m + '.html',
        idx, plot_bins, bin_labels,
        train_true_origin, val_true_origin, test_true_origin,
        err_train_origin, err_val_origin, err_test_origin,
        train_pred_origin, val_pred_origin, test_pred_origin, future_pred_origin,
    )

    # —— 相对误差, 2×2 ——
    fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    for j, bi in enumerate(plot_bins):
        ax = axs[j // 2, j % 2]
        ax.plot(idx[look_back:train_end], train_error[:, bi], 'b-', lw=0.5)
        ax.plot(idx[train_end:val_end],     val_error[:, bi], 'y-', lw=0.5)
        ax.plot(idx[val_end:test_end],     test_error[:, bi], 'm-', lw=0.5)
        ax.axhline(0, color='k', lw=0.5)
        ax.text(0.02, 0.88, bin_labels[j], transform=ax.transAxes, fontsize=11, fontweight='bold')

        split_style = dict(color='0.3', ls=':', lw=1.0, alpha=0.7, zorder=0)
        ax.axvline(idx[train_end], **split_style)
        ax.axvline(idx[val_end],   **split_style)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.96, bottom=0.08, wspace=0.06, hspace=0.10)
    fig.supylabel('Relative Error', x=0.03, fontsize=14)
    fig.supxlabel('Year', y=0.03, fontsize=14)
    plt.savefig('./Figure/lstmdraww/electron_error_' + m + '.pdf', bbox_inches='tight')
    plt.close()
    write_error_html(
        './Figure/lstmdraww/electron_error_' + m + '.html',
        idx, plot_bins, bin_labels, train_error, val_error, test_error,
    )

print("Done ✅")
