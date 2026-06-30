#!/bin/python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

def delta_days(date1, date2):
    # 将字符串转换为datetime对象
    d1 = datetime.strptime(date1, "%Y:%m:%d")
    d2 = datetime.strptime(date2, "%Y:%m:%d")
    # 计算时间差（返回天数）
    delta = abs(d2 - d1)
    return delta.days
def Plot(array):
    plt.plot(array)
    plt.show()
def Print(array):
    for i in range(len(array)):
       print(i, array[i])

# 近地磁场
BE = np.genfromtxt("../rawdata/omni_m_daily_latest.txt", usecols=8, dtype=float)
be_daily = BE[17306:22646] # from 2010.5.20 , 140st to 2024.12.31 
np.save("../sun_processed/latest/be_daily_latest.npy", be_daily)
print('be_daily ', be_daily.shape, '  ', be_daily[0], '', be_daily[-1]) #5340

# 太阳风速度
VSW = np.genfromtxt("../rawdata/omni_m_daily_latest.txt", usecols=9, dtype=float)
vsw_daily = VSW[17306:22646] #from 2010.5.20 , 140st to 2024.12,31
np.save("../sun_processed/latest/vsw_daily_latest.npy", vsw_daily)
print('vsw_daily ', vsw_daily.shape, '  ', vsw_daily[0],' ', vsw_daily[-1]) # 5340

# 倾角
TILT_DATE = np.genfromtxt("../rawdata/tilt_latest.txt", usecols=(2), dtype=(str))
TILT = np.genfromtxt("../rawdata/tilt_latest.txt", usecols=(4), dtype=(float))
combined = np.rec.fromarrays([TILT_DATE, TILT], names=['date', 'tilt'])
tilt_bartel = combined[456:653] # from 2010.5.19 to 2025.01.06
print(tilt_bartel.shape,'  ',tilt_bartel[0],'  ', tilt_bartel[-1]) # 198
xtilt_bartel = np.arange(len(tilt_bartel))
xtilt_daily = []
for i in range(len(xtilt_bartel) - 1):
    # 在 xtilt_bartel[i] 和 xtilt_bartel[i+1] 之间生成 n_interp 个点
    n_interp = delta_days(tilt_bartel[i]['date'], tilt_bartel[i+1]['date'])
    segment = np.linspace(xtilt_bartel[i], xtilt_bartel[i+1], num=n_interp, endpoint=False)
    xtilt_daily.extend(segment)
xtilt_daily.append(xtilt_bartel[-1])  # 添加最后一个原始点
# 转换为 NumPy 数组
xtilt_daily = np.array(xtilt_daily)
# 执行线性插值
tilt_daily = np.interp(xtilt_daily, xtilt_bartel, tilt_bartel[:]['tilt'])
tilt_daily = np.round(tilt_daily[1:1+5340],1) #from 2010.5.20 to 2024.12.31
np.save("../sun_processed/latest/tilt_daily_latest.npy", tilt_daily)
print('tilt_daily ', tilt_daily.shape)

# 太阳极化
SP = np.genfromtxt("../rawdata/mf_sun_polar_latest.txt", usecols=8, dtype=float)[1:]
print(SP[1342], SP[1343], SP[1593], SP[1594], SP[1690],SP[1691], SP[1692], SP[1693], SP[1694], SP[1695], SP[1708],SP[1726],SP[1727],SP[1728],SP[1729], SP[1741], SP[1742], SP[1743], SP[1744])
SP[1342]= -1.0
SP[1343]=  1.0
SP[1593] = 1.0
SP[1594] = 1.0
SP[1690] = 1.0 
SP[1691] = 1.0 
SP[1692] = 1.0 
SP[1693] = 1.0 
SP[1694] = 1.0 
SP[1695] = 1.0 
SP[1708] = 1.0 
SP[1726] = 1.0 
SP[1727] = 1.0 
SP[1728] = -1.0 
SP[1729] = -1.0 
SP[1741] = -1.0 
SP[1742] = -1.0 
SP[1743] = -1.0 
SP[1744] = -1.0 
sp = SP[1240:1776] # from 2010.5.13 to 2025.01.04
print('sp: ',sp.shape,' ', sp[0], sp[-1]) # 536
sp_tendays = np.sign(np.array(sp))
sp_daily = []
for i in range(len(sp_tendays)-1):
    sp_daily.extend([sp_tendays[i]]*10) 
# 转换为 NumPy 数组
sp_daily = np.array(sp_daily)
sp_daily = sp_daily[7:7+5340]
def sigmoid(x, k=0.03, x0=0):
    return 1 / (1 + np.exp(-k * (x - x0)))
xsp_daily = np.arange(len(sp_daily))
print(sp_daily[1022], sp_daily[1023], sp_daily[4872], sp_daily[4873])
sp_daily_sigmoid = -1+2*sigmoid(xsp_daily, k=0.03, x0=1022)-2*sigmoid(xsp_daily, k=0.03, x0=4872)
np.save("../sun_processed/latest/sp_daily_sigmoid_latest.npy", sp_daily_sigmoid)
print('sp_daily ', sp_daily_sigmoid.shape,'  ', sp[0],'  ', sp[-1])

# 太阳黑子数
SSN=pd.read_csv("../rawdata/SSN_d_tot_V2.0_latest.csv", sep=';')
ssn = SSN.values
ssn_daily = ssn[70265:75605,4] # # from 2010.5.20 to 2024.12.31
# 保存为numpy文件（保留数据类型）
np.save("../sun_processed/latest/ssn_daily_latest.npy", ssn_daily)
print('ssn_daily ', ssn_daily.shape,'  ', ssn_daily[0],'  ',ssn_daily[-1]) # 5340

Be_daily = np.reshape(be_daily,[-1,1])
Vsw_daily = np.reshape(vsw_daily,[-1,1])
Tilt_daily = np.reshape(tilt_daily,[-1,1])
Sp_daily = np.reshape(sp_daily_sigmoid,[-1,1])
Ssn_daily = np.reshape(ssn_daily,[-1,1])
sun4_daily = np.concatenate([Be_daily,Vsw_daily,Tilt_daily,Sp_daily],axis=1)
sun5_daily = np.concatenate([Be_daily,Vsw_daily,Tilt_daily,Sp_daily,Ssn_daily],axis=1)
np.save("../sun_processed/latest/sun4_daily_latest.npy", sun4_daily)
np.save("../sun_processed/latest/sun5_daily_latest.npy", sun5_daily) # 5340


# 包括理论计算2025-01-01 ~ 2031
SUN   = np.genfromtxt("../rawdata/para_solar_predict.txt", skip_header=1)
YEAR  = SUN[60:, 0].astype(int)
MONTH = SUN[60:, 1].astype(int)
SSN   = SUN[60:, 2]
SP    = SUN[60:, 3]
SP    = np.sign(np.array(SP))
TILT  = SUN[60:, 4]
VSW   = SUN[60:, 5]
BE    = SUN[60:, 6]
date = [f"{y}:{m:02d}:01" for y, m in zip(YEAR, MONTH)]
#print(YEAR)
#print(MONTH)
print(date)


xSSN = np.arange(len(SSN))
xSSN_daily = []
xTILT = np.arange(len(TILT))
xTILT_daily = []
xVSW = np.arange(len(VSW))
xVSW_daily = []
xBE = np.arange(len(BE))
xBE_daily = []
for i in range(len(date) - 1):
    n_interp = delta_days(date[i], date[i+1])
    segment_SSN = np.linspace(xSSN[i], xSSN[i+1], num=n_interp, endpoint=False)
    xSSN_daily.extend(segment_SSN)
    segment_TILT = np.linspace(xTILT[i], xTILT[i+1], num=n_interp, endpoint=False)
    xTILT_daily.extend(segment_TILT)
    segment_VSW = np.linspace(xVSW[i], xVSW[i+1], num=n_interp, endpoint=False)
    xVSW_daily.extend(segment_VSW)
    segment_BE = np.linspace(xBE[i], xBE[i+1], num=n_interp, endpoint=False)
    xBE_daily.extend(segment_BE)

xSSN_daily.append(xSSN[-1])  # 添加最后一个原始点
xSSN_daily = np.array(xSSN_daily)
SSN_daily = np.interp(xSSN_daily, xSSN, SSN)
np.save("../sun_processed/latest/ssn_daily_predict_latest.npy", SSN_daily)
print('SSN_daily ', SSN_daily.shape)


SP_daily = np.ones(2526) *(-1.0)
np.save("../sun_processed/latest/sp_daily_predict_latest.npy", SP_daily)
print('SP_daily ', SP_daily.shape)

xTILT_daily.append(xTILT[-1])  # 添加最后一个原始点
xTILT_daily = np.array(xTILT_daily)
TILT_daily = np.interp(xTILT_daily, xTILT, TILT)
np.save("../sun_processed/latest/tilt_daily_predict_latest.npy", TILT_daily)
print('TILT_daily ', TILT_daily.shape)

xVSW_daily.append(xVSW[-1])  # 添加最后一个原始点
xVSW_daily = np.array(xVSW_daily)
VSW_daily = np.interp(xVSW_daily, xVSW, VSW)
np.save("../sun_processed/latest/vsw_daily_predict_latest.npy", VSW_daily)
print('VSW_daily ', VSW_daily.shape)

xBE_daily.append(xBE[-1])  # 添加最后一个原始点
xBE_daily = np.array(xBE_daily)
BE_daily = np.interp(xBE_daily, xBE, BE)
np.save("../sun_processed/latest/be_daily_predict_latest.npy", BE_daily)
print('BE_daily ', BE_daily.shape)

BE_daily_predict   = np.reshape(BE_daily,  [-1,1])
VSW_daily_predict  = np.reshape(VSW_daily, [-1,1])
TILT_daily_predict = np.reshape(TILT_daily,[-1,1])
SP_daily_predict   = np.reshape(SP_daily,  [-1,1])
SSN_daily_predict  = np.reshape(SSN_daily, [-1,1])
sun4_daily_predict = np.concatenate([BE_daily_predict,
                                     VSW_daily_predict,
                                     TILT_daily_predict,
                                     SP_daily_predict],axis=1)
sun5_daily_predict = np.concatenate([BE_daily_predict,
                                     VSW_daily_predict,
                                     TILT_daily_predict,
                                     SP_daily_predict,
                                     SSN_daily_predict],axis=1)
np.save("../sun_processed/latest/sun4_daily_predict_latest.npy", sun4_daily_predict)
np.save("../sun_processed/latest/sun5_daily_predict_latest.npy", sun5_daily_predict) #2526


sun4_daily_all = np.concatenate([sun4_daily, sun4_daily_predict], axis=0)
sun5_daily_all = np.concatenate([sun5_daily, sun5_daily_predict], axis=0)

print(sun5_daily.shape, '  ', sun5_daily_predict.shape,'  ',sun5_daily_all.shape) #7866

np.save("../sun_processed/latest/sun4_daily_all_latest.npy", sun4_daily_all)
np.save("../sun_processed/latest/sun5_daily_all_latest.npy", sun5_daily_all)

import matplotlib.pyplot as plt
import numpy as np

# 示例数据
years = np.arange(2010, 2035)  # 年份范围
B_obs = np.random.rand(len(years)) * 20  # 磁场强度观测值
V_sw_obs = np.random.rand(len(years)) * 600 + 400  # 太阳风速度观测值
alpha_obs = np.random.rand(len(years)) * 75  # 角度观测值
A_obs = np.random.randint(-1, 2, len(years))  # 参数A观测值
SSN_obs = np.random.rand(len(years)) * 200  # 太阳黑子数观测值

B_pred = np.linspace(8, 7, len(years))  # 磁场强度预测值
V_sw_pred = np.linspace(400, 400, len(years))  # 太阳风速度预测值
alpha_pred = np.linspace(10, 10, len(years))  # 角度预测值
A_pred = np.linspace(-1, -1, len(years))  # 参数A预测值
SSN_pred = np.linspace(100, 0, len(years))  # 太阳黑子数预测值

# 创建图形和子图
fig, axs = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
idx = pd.date_range(start="2010-05-20", end="2031-12-1")

# 设置标题
fig.suptitle('Daily solar activity', fontsize=16)

# 绘制磁场强度
axs[0].plot(idx[0:5340], sun5_daily[:,0], 'g-', label='Observation')
axs[0].plot(idx[5340:],  sun5_daily_predict[:,0], 'r--', label='Theory Prediction')
axs[0].set_ylabel('B (nT)')
axs[0].legend(loc='upper right')

# 绘制太阳风速度
axs[1].plot(idx[0:5340], sun5_daily[:,1], 'g-')
axs[1].plot(idx[5340:], sun5_daily_predict[:,1], 'r--')
axs[1].set_ylabel('V_sw (km/s)')

# 绘制角度
axs[2].plot(idx[0:5340], sun5_daily[:,2], 'g-')
axs[2].plot(idx[5340:], sun5_daily_predict[:,2], 'r--')
axs[2].set_ylabel('α (degree)')

# 绘制参数A
axs[3].plot(idx[0:5340], sun5_daily[:,3], 'g-')
axs[3].plot(idx[5340:], sun5_daily_predict[:,3], 'r--')
axs[3].set_ylabel('A')

# 绘制太阳黑子数
axs[4].plot(idx[0:5340], sun5_daily[:,4], 'g-')
axs[4].plot(idx[5340:], sun5_daily_predict[:,4], 'r--')
axs[4].set_ylabel('SSN')
axs[4].set_xlabel('Year')

# 调整布局以避免重叠
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 显示图表
plt.show()
plt.savefig('./Figure/data/sun5_latest_24.pdf')





