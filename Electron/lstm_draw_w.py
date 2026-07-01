#!/bin/python
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


import os
os.makedirs('./Data/lstm2', exist_ok=True)
os.makedirs('./Figure/lstm2', exist_ok=True)

# =====================================================
# 1. 载入太阳和流强数据
# =====================================================
sun_daily      = np.load('../sun_processed/latest/sun5_daily_all_latest.npy')
electron_daily = np.load('Data/flux/electron_flux_allbin.npy')
electron_error_daily = np.load('Data/flux/electron_flux_abs_error_allbin.npy')

print('sun daily all latest : ', sun_daily.shape)
print('electron flux daily : ',  electron_daily.shape)
print('electron error daily : ',  electron_error_daily.shape)

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
n_features = 5 + 2*bins
train_num  = 0.6
val_num    = 0.2
print('number = ', number, 'bins = ', bins, 'sun_offset =', sun_offset)

# 只组合观测数据
Series = np.concatenate([sun_daily[sun_offset:sun_offset+number], electron_daily, electron_error_daily], axis=1)
print('Series = ', Series.shape)

# =====================================================
# 2. 划分训练、验证和测试集
# =====================================================
train_end = round(number * train_num)
val_end   = round(number * (train_num + val_num))
test_end  = number
future_end = sun_daily.shape[0] - sun_offset
print('train_end, val_end, test_end, future_end:', train_end, val_end, test_end, future_end)

X_train = Series[0:train_end, :]
y_train = Series[0:train_end, 5:5+bins]
X_val   = Series[train_end-look_back:val_end, :]
y_val   = Series[train_end-look_back:val_end, 5:5+bins]
X_test  = Series[val_end-look_back:test_end, :]
y_test  = Series[val_end-look_back:test_end, 5:5+bins]

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

scaler_flux = MinMaxScaler()
y_train = scaler_flux.fit_transform(y_train)
y_val   = scaler_flux.transform(y_val)
y_test  = scaler_flux.transform(y_test)

# =====================================================
# 5. 填充未来待预测数据
# =====================================================
# 未来天数 = sun 总长 - (sun 偏移 + electron 天数)
future_days = sun_daily.shape[0] - (sun_offset + number)
future_flux = np.concatenate([Series[-look_back:, 5:5+bins], np.zeros([future_days, bins])], axis=0)
future_error = np.concatenate([Series[-look_back:, 5+bins:], np.zeros([future_days, bins])], axis=0)
padding_tail = np.concatenate([sun_daily[sun_offset+number-look_back:, 0:5], future_flux, future_error], axis=1)
padding_tail_scaled = scaler.transform(padding_tail)
X_future = padding_tail_scaled
y_future = padding_tail_scaled[:, 5:5+bins]

Series = np.concatenate([X_train, X_val[look_back:, :], X_test[look_back:, :], X_future[look_back:, :]], axis=0)

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

X_train_seq, y_train_seq = make_sequence(X_train, y_train, look_back)
X_val_seq,   y_val_seq   = make_sequence(X_val,   y_val,   look_back)
X_test_seq,  y_test_seq  = make_sequence(X_test,  y_test,  look_back)
print(X_train_seq.shape, y_train_seq.shape,
      X_val_seq.shape,   y_val_seq.shape,
      X_test_seq.shape,  y_test_seq.shape)

# =====================================================
# 7. 模型列表 — 填入最佳模型
# =====================================================
model_list = [
    # '0-5000epoch_0.0001learningRate_64neurons_0.002l2_0.08dropout_64batchSize_0217-0.00534.keras',
    # 'errWeighted_0-5000epoch_0.0001learningRate_64neurons_0.002l2_0.08dropout_64batchSize_0553-0.00321.keras',
]

if not model_list:
    model_list = [
        p.name for p in sorted(
            Path('./Data/model').glob('errWeighted_*.keras'),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:1]
    ]

if not model_list:
    raise FileNotFoundError('No errWeighted_*.keras model found in ./Data/model')

from tensorflow.keras.models import load_model

# =====================================================
# 8. 逐模型预测 & 评估
# =====================================================
for m in model_list:
    model = load_model('./Data/model/' + m, compile=False)
    if model.input_shape[-1] != n_features:
        raise ValueError(
            f'Model {m} expects {model.input_shape[-1]} features, '
            f'but current err-weighted workflow builds {n_features}.'
        )

    # --- 训练集预测 ---
    train_pred_mean   = model.predict(X_train_seq)
    train_pred_origin = scaler_flux.inverse_transform(train_pred_mean)
    train_true_origin = scaler_flux.inverse_transform(y_train[:, :])
    train_rme = np.mean(np.abs(train_pred_origin - train_true_origin[look_back:]) / train_true_origin[look_back:])

    # --- 验证集预测 ---
    val_pred_mean   = model.predict(X_val_seq)
    val_pred_origin = scaler_flux.inverse_transform(val_pred_mean)
    val_true_origin = scaler_flux.inverse_transform(y_val[look_back:, :])
    val_rme = np.mean(np.abs(val_pred_origin - val_true_origin) / val_true_origin)

    # --- 测试集预测 ---
    test_pred_mean   = model.predict(X_test_seq)
    test_pred_origin = scaler_flux.inverse_transform(test_pred_mean)
    test_true_origin = scaler_flux.inverse_transform(y_test[look_back:, :])
    test_rme = np.mean(np.abs(test_pred_origin - test_true_origin) / test_true_origin)

    print('mean relative error : train', train_rme, ', validation', val_rme, ', testing', test_rme)

    # --- 计算各段相对误差 ---
    train_error = np.array(train_pred_origin) / np.array(train_true_origin[look_back:]) - 1
    val_error   = np.array(val_pred_origin)   / np.array(val_true_origin) - 1
    test_error  = np.array(test_pred_origin)  / np.array(test_true_origin) - 1
    print('train error shape =', train_error.shape)

    np.save('./Data/lstm2/train_error_allbin_' + m + '.npy', train_error)
    np.save('./Data/lstm2/val_error_allbin_'   + m + '.npy', val_error)
    np.save('./Data/lstm2/test_error_allbin_'  + m + '.npy', test_error)

    # --- 未来预测 (滑动窗口自回归) ---
    init_batch = Series[number-look_back:number, :n_features].reshape((1, look_back, n_features))
    cur_batch = init_batch

    future_pred_mean = []
    for i in range(future_end - number):
        if i % 100 == 0:
            print('future step', i, '/', future_end - number)
        future_pred = model.predict(cur_batch, verbose=0)[0]
        Series[number + i, 5:5+bins] = future_pred
        future_pred_mean.append(future_pred)
        start = number - look_back + i + 1
        end   = number + i + 1
        cur_batch = Series[start:end, :n_features].reshape(1, look_back, n_features)

    future_pred_origin = scaler_flux.inverse_transform(future_pred_mean)

    # =====================================================
    # 9. 画图 — 4 个代表能档: ~1, 2, 5, 10 GeV, 2×2 布局
    # =====================================================
    # 加载真实能量边界
    energy_edges = np.load('Data/flux/electron_energy_edges.npy')  # (42,) for 41 bins
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
        ax.plot(idx[look_back:train_end], train_true_origin[look_back:, bi], 'g-', lw=0.8)
        ax.plot(idx[train_end:val_end],     val_true_origin[:, bi], 'g-', lw=0.8)
        ax.plot(idx[val_end:test_end],     test_true_origin[:, bi], 'g-', lw=0.8)
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
            ax.text(idx[pos_test],  0.03, 'testing',    transform=trans, ha='center', va='bottom', fontsize=10)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.96, bottom=0.08, wspace=0.06, hspace=0.10)
    fig.supylabel('Electron Flux  [m$^{-2}$ s$^{-1}$ sr$^{-1}$ (GeV/n)$^{-1}$]', x=0.03, fontsize=14)
    fig.supxlabel('Year', y=0.03, fontsize=14)
    plt.savefig('./Figure/lstm2/electron_prediction_' + m + '.pdf', bbox_inches='tight')
    plt.close()

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
    plt.savefig('./Figure/lstm2/electron_error_' + m + '.pdf', bbox_inches='tight')
    plt.close()

print("Done ✅")
