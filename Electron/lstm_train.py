# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
import argparse
import os


parser = argparse.ArgumentParser(description="Train a model with specified yaml file.")
parser.add_argument('--config', type=str, default='config.yaml',
                   help='Path to the YAML config file (default: config.yaml)')

args = parser.parse_args()

with open(args.config, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print('config: electron ',
            config['epoch_begin'],'epoch_begin ',
            config['epochs'],'epochs ',
            config['learning_rate'],'learning_rate ',
            config['batch_size'],'batch_size ',
            config['neurons'],'neurons ',
            config['l2'],'l2 ',
            config['dropout'],'dropout ',
            config['train_num'],'train_num',
            config['val_num'],'val_num',
            config['look_back'],'look_back '
            )

# set the random seed, to reproduce the same results for each computation with the same code
look_back = config['look_back']
train_num = config['train_num']
val_num   = config['val_num']
neurons  = config['neurons']
l2  = config['l2']
dropout   = config['dropout']


# 载入太阳和流强数据
sun_daily      = np.load('../sun_processed/latest/sun5_daily_all_latest.npy')
electron_daily = np.load('Data/flux/electron_flux_allbin.npy')

print('sun daily all latest : ', sun_daily.shape)
print('electron flux daily : ',  electron_daily.shape)

# 电子前补 365 天零 → 让 sun 参数更早进入历史窗口
pad_days  = 365
electron_daily = np.concatenate([np.zeros([pad_days, electron_daily.shape[1]]), electron_daily])

# 自动计算 sun offset: sun 从 2010-05-20 起, 补零后电子从 2010-06-11 起
flux_start = pd.Timestamp('2011-06-11') - pd.Timedelta(days=pad_days)
sun_start  = pd.Timestamp('2010-05-20')
sun_offset = (flux_start - sun_start).days  # 22

number = electron_daily.shape[0]
bins   = electron_daily.shape[1]
print('number = ', number, 'bins = ', bins, 'sun_offset =', sun_offset)
# 只组合观测数据(number, 5+bins)
Series = np.concatenate([sun_daily[sun_offset:sun_offset+number], electron_daily], axis=1)
print('Series = ', Series.shape)

# 定义训练、验证和测试集 ---------------|---|---|
train_end = int(number*train_num)
val_end = int(number*(train_num+val_num))
test_end = number
future_end = sun_daily.shape[0]
print(train_end, val_end, test_end, future_end)

# 后边的训练数据需要加上上一年的太阳数据
X_train = Series[0:train_end,      0:]  # (0, 70%)
y_train = Series[0:train_end,      5:]
X_val = Series[train_end-look_back:val_end,  0:] # (70%, 85%)
y_val = Series[train_end-look_back:val_end,  5:]
X_test =  Series[val_end-look_back:test_end, 0:] # (85%, 1)
y_test =  Series[val_end-look_back:test_end, 5:]

seed = 42
import random
import tensorflow as tf
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 归一化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val) # 只缩放
X_test = scaler.transform(X_test) # 只缩放

scaler_flux = MinMaxScaler()
y_train = scaler_flux.fit_transform(y_train)
y_val = scaler_flux.transform(y_val)
y_test = scaler_flux.transform(y_test)

print(np.isnan(X_train).any(), np.isinf(X_train).any())
print(np.isnan(y_train).any(), np.isinf(y_train).any())
print(np.isnan(X_val).any(), np.isinf(X_val).any()    )
print(np.isnan(y_val).any(), np.isinf(y_val).any()    )


# ================== 序列化函数 ==================
def make_sequence(X, y, look_back):
    X_seq, y_seq = [], []
    for i in range(look_back, len(X)):
        X_seq.append(X[i - look_back:i, :])   # 过去 look_back 天特征
        y_seq.append(y[i, :])                 # 预测当天的 flux
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = make_sequence(X_train, y_train, look_back)
X_val_seq,   y_val_seq   = make_sequence(X_val,   y_val,   look_back)
X_test_seq,  y_test_seq  = make_sequence(X_test,  y_test,  look_back)
print(X_train_seq.shape, y_train_seq.shape)
# → (样本数, look_back, 6) , (样本数,), (样本数,)


import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.regularizers import L1L2, L1, L2
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Reshape
from tensorflow.keras.layers import BatchNormalization, Dropout

learning_rate = config['learning_rate']
batch_size    = config['batch_size']
epoch_begin= config['epoch_begin']
epochs= config['epochs']
epoch_end=epoch_begin+epochs

model = Sequential([
    Input(shape=(look_back, 5+bins), dtype='float32'),
    LSTM(units               = neurons, dropout=dropout,
        kernel_regularizer  = L1L2(l1=0, l2=l2),
        name                = 'LSTM', return_sequences=False  ),
    #Dense(64,  activation='relu'),
    Dense(bins,  name='output_flux')
    ])

adamax = Adamax(learning_rate = learning_rate, 
        beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07)
model.compile(optimizer=adamax, loss='huber', metrics=[], weighted_metrics=[])
#model.compile(optimizer=adamax, loss='huber')
model.summary()

from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, Callback, EarlyStopping

os.makedirs('./Data/model', exist_ok=True)
os.makedirs('./Figure/lstmtrain', exist_ok=True)

checkpoint = ModelCheckpoint('./Data/model/'
        +str(epoch_begin)+'-'
        +str(epoch_end)+'epoch_'
        +str(learning_rate)+'learningRate_'
        +str(neurons)+'neurons_'
        +str(l2)+'l2_'
        +str(dropout)+'dropout_'
        +str(batch_size)+'batchSize_'
        +'{epoch:04d}-{val_loss:.5f}'
        +'.keras',
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='auto')

class TestLossCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_test, y_test_true, delta=1.0):
        super().__init__()
        self.X_test = X_test
        self.y_test_true = tf.constant(y_test_true, dtype=tf.float32)
        self.delta = delta
        self.loss = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_test, verbose=0)
        y_pred = tf.constant(y_pred, dtype=tf.float32)

        # 手动计算逐通道 Huber loss
        abs_error = tf.abs(self.y_test_true - y_pred)
        quadratic = tf.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * tf.square(quadratic) + self.delta * linear  # (batch, 8)
        test_loss = tf.reduce_mean(loss).numpy()

        # MSE: (y_true - y_pred)^2
        #mse = tf.square(self.y_test_true - y_pred)   # (batch, channels)
        #test_loss = tf.reduce_mean(mse).numpy()



        self.loss.append(test_loss)
        logs['test_loss'] = test_loss


test_callback = TestLossCallback(X_test_seq, y_test_seq)


early_stop = EarlyStopping(
        monitor='val_loss',           # 监控的指标
        patience=100,                 # 等待10轮没有改善就停止
        min_delta=0.00001,             # 损失下降小于该值视为无改善
        mode='min',                   # 目标是最小化 val_loss
        verbose=1                    # 打印停止信息
        )



history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs= epochs,
        batch_size= batch_size,
        callbacks=[checkpoint, test_callback, early_stop],
        verbose=1
        )


A=0
plt.plot(history.history['loss'][A:])
plt.plot(history.history['val_loss'][A:])
plt.plot(test_callback.loss[A:])
plt.yscale('log')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation', 'test'], loc='upper left')
plt.savefig('./Figure/lstmtrain/loss_'
        +str(epoch_begin)+'-'
        +str(epoch_end)+'epoch_'
        +str(neurons)+'neurons_'
        +str(l2)+'l2_'
        +str(dropout)+'dropout_'
        +str(learning_rate)+'learningRate_'
        +str(batch_size)+'batchSize'
        +'.pdf', bbox_inches='tight')
plt.close()
