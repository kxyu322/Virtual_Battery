import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

ON = True
OFF = False

CIDX = 0            # 电池容量的index
P_chargeIDX = 1     # 充电功率的～
P_dischargeIDX = 2  # 放电功率的～
alphaIDX = 3        # 耗散系数的～

NumTCL = 1000       # 空调的数量
t_span = 3600       # 仿真的时长/s
h = 0.1             # 仿真的时间步长/s
Amplifier = 500    # 对AGC信号（-1到1）的放大/kW
ONOFF = np.random.choice([ON, OFF], size=NumTCL)    # 在某一时刻空调集群的开关状态

CthIDX, Cth = 0, 2          # 热容：kWh/Celsus
RthIDX, Rth = 1, 2          # 热阻：Celsus/kW
PmIDX,  Pm  = 2, 5.6        # 工作功率：kW
etaIDX, eta = 3, 2.5        # 电-热转换效率：none
TrIDX,  Tr  = 4, 22.5       # 设定温度：Celsus
TdeltaIDX, Tdelta = 5, 0.3  # 温度死区
PoIDX = 6                   # 平均功率的index
Ta = 32                     # 环境温度：Celsus
# T0 = np.random.uniform(low=0.95*Tr, high=1.05*Tr, size=NumTCL) # 按照高斯初始化房间温度

para = []  # 全局变量，存储6个参数

def prepare_tcl_data_for_echarts(*args):
    """
    准备空调监控数据以供ECharts使用
    *args: 需要展示的空调编号。如1,2,3表示展示这3台空调的数据
    返回: 适用于ECharts的JSON格式数据
    """
    result = []
    
    for idx in args:
        # 获取时间点
        time_points = ts[::int(4/h)].tolist()
        
        # 获取温度数据
        temp_data = Ts[::int(4/h), idx].tolist()
        
        # 获取温度上下限
        upper_band = float(para[TrIDX][idx] + para[TdeltaIDX][idx]/2)
        down_band = float(para[TrIDX][idx] - para[TdeltaIDX][idx]/2)
        
        # 获取开关状态数据
        onoff_data = ONOFFs[:,idx].tolist()
        
        # 添加到结果中
        tcl_info = {
            "id": int(idx),
            "timePoints": time_points,
            "temperatureData": temp_data,
            "upperBand": upper_band,
            "lowerBand": down_band,
            "onOffData": onoff_data
        }
        
        result.append(tcl_info)
    
    return result

def dTheta(q, Tt, a, b, Pm, Ta=Ta):
    return -a * (Tt - Ta) - (b * Pm if q == ON else 0)

def ParaCreate(Cth=Cth, Rth=Rth, Pm=Pm, eta=eta, Tr=Tr, Tdelta=Tdelta):
    global T0

    # print(Cth, Rth, Pm, eta, Tr, Tdelta)
    # 单个空调的6个参数
    params = [Cth, Rth, Pm, eta, Tr, Tdelta]

    # params为中心值，正态分布地创建NumTCL组参数
    Paras = [np.random.uniform(low=0.95*para, high=1.05*para, size=NumTCL) for para in params]

    # 计算平均功率Po
    Ton = Paras[RthIDX] * Paras[CthIDX] * np.log((Paras[TrIDX]+Paras[TdeltaIDX]-Ta+Paras[RthIDX]*Paras[PmIDX]*Paras[etaIDX])\
                                                  /(Paras[TrIDX]-Paras[TdeltaIDX]-Ta+Paras[RthIDX]*Paras[PmIDX]*Paras[etaIDX]))
    Toff = Paras[RthIDX] * Paras[CthIDX] * np.log((Paras[TrIDX]-Paras[TdeltaIDX]-Ta)\
                                                   /(Paras[TrIDX]+Paras[TdeltaIDX]-Ta))
    array = Paras[PmIDX] * Ton / (Ton + Toff)
    Paras.append(array)

    T0 = Paras[TrIDX] # 房间温度从设定温度开始，保持模型稳定

    return Paras

def euler_step_temp(y, para, n):
    Rth = para[RthIDX][n]
    Cth = para[CthIDX][n]
    eta = para[etaIDX][n]
    Pm = para[PmIDX][n]
    
    # a和b的定义在之前讲过
    a = 1/(Rth * Cth * 3600)
    b = eta / (Cth * 3600)
    
    return y + h * dTheta(q=ONOFF[n], Tt=y, a=a, b=b, Pm=Pm) # 欧拉法的微分方程

def ImportAGC():
    file_path = './virtual_battery/data/rto-regulation-signal-08-2024.xlsx'
    sheet_name = 'Dynamic'
    data = pd.read_excel(file_path, sheet_name)

    # 取“第1列（去掉时间列），所有行”的调节信号
    # 把信号放大到Amplifier数量级
    return np.array([data * Amplifier for data in data.iloc[1000:, 1].to_list()])

def BatteryModel(para):
    # 必要模型
    # 充放电功率
    P_discharge =  -sum(para[PoIDX])
    P_charge = (sum(para[PmIDX]) - sum(para[PoIDX]))  
    
    # 容量
    Rth = para[RthIDX]
    Cth = para[CthIDX]
    eta = para[etaIDX]
    T_delta = para[TdeltaIDX]
    a = 1/(Rth * Cth * 3600)
    b = eta / (Cth * 3600)
    alpha = np.average(a)
    C = sum((1 + abs(1 - a/alpha))*T_delta/b) / 3600    # a和b中的时间单位为s，为使C为kwh，故/3600

    # 必要模型的4个参数
    B_necessary = [C, P_charge, P_discharge, alpha]     

    # 使放电功率n_-最大的充分模型
    # 充放电功率
    P_discharge = -sum(para[PoIDX])
    P_charge = -P_discharge * min((para[PmIDX] - para[PoIDX])/para[PoIDX])   

    # 容量
    f = T_delta/(b*(1 + abs(alpha/a - 1)))
    C = -P_discharge * min(f/para[PoIDX]) / 3600        # 从kWs转换到kWh

    # 充分模型的4个参数
    # P_charge = np.float64(2000.0)   # 必要模型的充电功率算出来和论文不一致，故手动设为一致
    B_sufficient = [C, P_charge, P_discharge, alpha]

    return B_necessary, B_sufficient

def init_var():
    global ts, Ts, Ps, ONOFFs
    ts = np.arange(start=0, stop=t_span, step=h)    # 仿真时长/仿真步长=仿真步数
    Ts = np.zeros(shape=[len(ts), NumTCL])          # Ts[i,j]表示第j台空调在第i个时间步时的温度
    Ps = np.zeros(shape=t_span // 4)                # Ps[i]表示第4*i秒时调节信号与空调集群偏移功率的差值
    ONOFFs = np.tile(ONOFF, (len(Ps), 1))           # ONOFFs[i,j]表示第i台空调在第4*j秒时的开关状态
    Ts[0] = T0                                      # 空调集群初始温度

def control(i, control_idx, agc, para, ONOFF):
    """
    i（int）：在第i个时间步进行控制
    control_idx（int）：Ps数组中的index。
    agc：第i个时间步的控制信号。AGC数组数据之间相隔2s
    para：6个参数
    ONOFF：第i-1时间步，空调集群的开关状态
    """
    # 使用必要模型控制，满足每个空调的温度要求
    within_band = np.array((Ts[i - 1] >= para[TrIDX] - para[TdeltaIDX]/2) & 
                           (Ts[i - 1] <= para[TrIDX] + para[TdeltaIDX]/2))  # 每台空调的温度是否在范围内
    above_band = np.array(Ts[i-1] > para[TrIDX] + para[TdeltaIDX]/2)        # 每台空调的温度是否超过上限
    below_band = np.array(Ts[i-1] < para[TrIDX] - para[TdeltaIDX]/2)        # 每台空调的温度是否超过下限

    # 初步控制：改变把温度过限的空调状态
    # 这些空调和band的状态没有改变，因此不会参与下面的stack
    ONOFF[above_band] = ON      # 把温度过高的空调打开
    ONOFF[below_band] = OFF     # 把温度过低的空调关闭

    # 计算虚拟电池的充电功率
    P_delta = sum(para[PmIDX][ONOFF == ON]) - sum(para[PoIDX])

    # 建立一个stack
    if P_delta <= agc:
        # 增加充电功率，需要建立关闭的空调的stack
        charge = True      
        mask = (ONOFF == OFF) & within_band                     # 在满足温度范围的空调群内寻找关闭的空调
        Tdiff = (para[TrIDX] + para[TdeltaIDX]/2 - Ts[i - 1])\
                /para[TdeltaIDX]                                # 温度上限-空调温度，越小说明越紧迫
        candidates = np.where(mask, Tdiff, +np.inf)            # 把非目标空调的值变最大，即最不紧迫
        sorted_indices = np.argsort(candidates)                 # 从小到大排序
    else:
        # 减少充电功率，需要建立打开的空调的stack
        charge = False
        mask = (ONOFF == ON) & within_band                      # 在满足温度范围的空调群内寻找关闭的空调
        Tdiff = (Ts[i - 1] - (para[TrIDX] - para[TdeltaIDX]/2))\
                /para[TdeltaIDX]                                # 空调温度-温度下限，越小说明越紧迫
        candidates = np.where(mask, Tdiff, +np.inf)            # 把非目标空调的值变最大，即最不紧迫
        sorted_indices = np.argsort(candidates)                 # 从小到大排序

    # 根据stack来开关空调，直到P_delta与agc的大小关系变化
    for idx in sorted_indices:
        if mask[idx] == False:  # 跳过不在控制范围内的空调
            continue
        P_delta += para[PmIDX][idx] * 1 if charge == True else -1   # 打开或关闭这台空调
        ONOFF[idx] = ON if charge == True else OFF                  # 空调状态取反
        stop = P_delta >= agc if charge == True else P_delta <= agc # 当P_delta满足agc的要求
        if stop:
            break

    Ps[control_idx] = agc - P_delta     # AGC信号与空调集群偏移功率的差值，control_idx*4*h=实际时间
    ONOFFs[control_idx, :] = ONOFF      # 增加空调集群在这一个时间步的开关状态


def simulate():
    for i in tqdm(range(1, len(ts))):   # ts表示仿真步数
        current_time = i * h            # i*h表示实际时间

        # 每4s进行一次控制
        if i % int(4/h) == 0:
            control_idx = int(current_time // 4)    # control_idx用于Ps, *4就是实际时间
            agc_idx = int(current_time // 2)        # agc_idx用于AGC。AGC本身2s为间隔。
            control(i, control_idx, AGC[agc_idx], para, ONOFF)

        # 基于欧拉法更新空调集群的温度
        for n in range(NumTCL):
            Ts[i, n] = euler_step_temp(Ts[i-1, n], para, n)
        # 基于欧拉法更新电池荷电状态
        

def plot_power_and_soc(ts, h, AGC, Ps, Bn, Bs, t_span, initial_soc=0.5):
    """
    ts, h           : 原仿真时间向量和步长
    AGC             : 导入的调节信号数组
    Ps              : 电池的实际功率偏差，len = t_span//4
    Bn, Bs          : BatteryModel() 返回的必要/充分模型参数列表
    t_span          : 总仿真时长（s）
    initial_soc     : 初始 SOC（0~1），默认 50%
    """
    delta_t = 4
    step    = int(delta_t / h)
    # 对齐横轴
    time_plot = ts[::step]
    # 调节信号与功率偏差
    P_reg = AGC[0 : t_span//2 : 2]
    P_dev = P_reg - Ps

    # —— 上子图：功率曲线 ——  
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,8))
    ax1.plot(time_plot, P_reg, label="Regulation Signal", color='blue')
    ax1.plot(time_plot, P_dev, label="Power Deviation", color='orange', linestyle='--')
    # 必要/充分限值
    ax1.hlines([Bn[P_chargeIDX], Bn[P_dischargeIDX]],
               xmin=time_plot[0], xmax=time_plot[-1],
               colors='black', linestyles='-.',
               label="Necessary Limits")
    ax1.hlines([Bs[P_chargeIDX], Bs[P_dischargeIDX]],
               xmin=time_plot[0], xmax=time_plot[-1],
               colors='green', linestyles='-.',
               label="Sufficient Limits")
    ax1.set_ylabel("Power (kW)")
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # —— 下子图：SOC 曲线 ——  
    # 先积分算 SOC（kWh）
    C = Bs[CIDX]
    N = len(P_dev)
    soc = np.zeros(N, dtype=float)
    soc[0] = 0
    for i in range(1, N):
        # 放电(P_dev>0) SOC 减少；充电(P_dev<0) SOC 增加
        soc[i] = soc[i-1] - P_dev[i] * (delta_t / 3600.0)
        
    ax2.plot(time_plot, soc, label="SOC (%)", color='purple')
    # 必要/充分限值
    ax2.hlines([Bn[CIDX], -Bn[CIDX]],
               xmin=time_plot[0], xmax=time_plot[-1],
               colors='black', linestyles='-.',
               label="Necessary Capacities")
    ax2.hlines([Bs[CIDX], -Bs[CIDX]],
               xmin=time_plot[0], xmax=time_plot[-1],
               colors='green', linestyles='-.',
               label="Sufficient Capacities")
    ax2.set_ylabel("Power (kW)")
    ax2.legend(loc='upper right')
    ax2.grid(True)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Energy(kWh)")
    ax2.legend(loc='upper right')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_TCLs_power(ts, h, AGC, Ps, t_span, PoSUM):
    """
    ts, h           : 原仿真时间向量和步长
    AGC             : 导入的调节信号数组
    Ps              : 电池的实际功率偏差，len = t_span//4
    t_span          : 总仿真时长（s）
    PoSUM           ; 集群的总基础功率
    """
    delta_t = 4
    step    = int(delta_t / h)
    # 对齐横轴
    time_plot = ts[::step]
    # 集群实际消耗功率
    P_reg = AGC[0 : t_span//2 : 2]
    P_real = P_reg - Ps + PoSUM

    # plot 
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(time_plot, P_real, label="Power drawn by 1000 ACs", color='orange')
    ax1.set_ylabel("Power (kW)")
    ax1.legend(loc='upper right')
    ax1.grid(True)


def TCL_plot(*args):
    """
    *args       :需要展示的空调编号。*args=1,2,3表示展示这3台空调的数据
    """
    num_plot = len(args)
    fig, axes = plt.subplots(2, num_plot, figsize=(8, 6))
    for i, idx in enumerate(args):
        temp = axes[0, i] if num_plot > 1 else axes[0]
        onoff = axes[1, i] if num_plot > 1 else axes[1]

        # plot temperature
        upper_band = para[TrIDX][idx]+para[TdeltaIDX][idx]/2
        down_band = para[TrIDX][idx]-para[TdeltaIDX][idx]/2
        temp.plot(ts[::int(4/h)], Ts[::int(4/h), idx], label=f"Temperature of {idx}")
        temp.plot(ts[::int(4/h)], [upper_band]*len(ts[::int(4/h)]), label=f"Upper {idx} = {upper_band:.1f}", color='red')
        temp.plot(ts[::int(4/h)], [down_band]*len(ts[::int(4/h)]), label=f"Down {idx} = {down_band:.1f}", color='green')
        temp.set_xlabel('Time (seconds)')
        temp.set_ylabel('Temperature (Celsus)')
        temp.grid(True)
        temp.legend()
        
        # plot ONOFF
        onoff.step(ts[::int(4/h)], ONOFFs[:,idx], label=f"ONOFF state of {idx}", color='orange')
        onoff.set_xlabel('Time (seconds)')
        onoff.set_ylabel('ONOFF')
        onoff.grid(True)
        onoff.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    global AGC, Bn, Bs

    para = ParaCreate()         # 得到6个参数
    init_var()                  # 初始化其他参数
    AGC = ImportAGC()           # 导入调节信号
    # AGC = [-50]*len(Ps)*2     # 直流测试信号
    Bn, Bs = BatteryModel(para) # 计算虚拟电池模型参数
    simulate()                  # 进行仿真

    # # 画图
    # plot_power_and_soc(ts, h, AGC, Ps, Bn, Bs, t_span, initial_soc=0.5)

    TCL_plot(0)

    plot_power_and_soc(ts, h, AGC, Ps, Bn, Bs, t_span, initial_soc=0.5)

    plot_TCLs_power(ts, h, AGC, Ps, t_span, sum(para[PoIDX]))

