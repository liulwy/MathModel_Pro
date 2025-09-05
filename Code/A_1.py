# -*- coding: utf-8 -*-
import json
import math
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 配置中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# ===== 问题一参数配置 =====
# 坐标系：假目标为原点(0,0,0)
# 真目标代表点（下底面圆心）
T = np.array([0.0, 200.0, 0.0], dtype=float)

# 来袭导弹 M1
M0 = np.array([20000.0, 0.0, 2000.0], dtype=float)  # 初始位置
vM = 300.0  # 速度(m/s)
uM = -M0 / np.linalg.norm(M0)  # 单位方向向量

# 无人机 FY1
F0 = np.array([17800.0, 0.0, 1800.0], dtype=float)  # 初始位置
vF = 120.0  # 飞行速度(m/s)
# 朝向假目标方向飞行（单位方向向量）
uF = -np.array([F0[0], F0[1], 0]) / np.linalg.norm([F0[0], F0[1], 0])

# 干扰弹参数
t_release = 1.5  # 投放时间(s)
delta_t = 3.6  # 投放后起爆延迟(s)
t_detonate = t_release + delta_t  # 起爆时间(s)

# 云团参数
v_sink = 3.0  # 云团下沉速度(m/s)
r_cloud = 10.0  # 云团有效半径(m)
t_window = 20.0  # 云团有效时间(s)

# 重力加速度
g = 9.8  # m/s²

# ===== 运动方程定义 =====
def missile_position(t: float) -> np.ndarray:
    """导弹位置方程"""
    return M0 + vM * uM * t

def drone_position(t: float) -> np.ndarray:
    """无人机位置方程"""
    return F0 + vF * uF * t

def bomb_position(t: float) -> np.ndarray:
    """干扰弹位置方程"""
    if t < t_release:
        return drone_position(t)
    
    # 投放后的抛体运动
    dt = t - t_release
    rel_pos = drone_position(t_release)
    return rel_pos + vF * uF * dt + np.array([0, 0, -0.5 * g * dt**2])

def cloud_center_position(t: float) -> np.ndarray:
    """云团中心位置方程"""
    det_pos = bomb_position(t_detonate)
    if t < t_detonate:
        return np.array([np.nan, np.nan, np.nan])
    
    # 起爆后匀速下沉
    dt = t - t_detonate
    return det_pos + np.array([0, 0, -v_sink * dt])

# ===== 几何计算函数 =====
def los_distance_sq_to_cloud_center(t: float) -> float:
    """计算导弹-真目标视线到云团中心的最小距离平方"""
    cloud_center = cloud_center_position(t)
    if any(np.isnan(cloud_center)):
        return np.inf
    
    # 导弹位置
    M_t = missile_position(t)
    
    # 真目标位置T到导弹位置M_t的向量
    TM = M_t - T
    
    # 云团中心到真目标的向量
    CT = cloud_center - T
    
    # 计算投影系数s
    s = np.dot(CT, TM) / np.dot(TM, TM)
    s_clamped = max(0.0, min(1.0, s))
    
    # 计算线段上离云团最近的点
    closest_point = T + s_clamped * TM
    
    # 返回距离平方
    return np.sum((closest_point - cloud_center)**2)

# ===== 二分法求根 =====
def bisection_root(f, a, b, tol=1e-6, max_iter=100):
    """在区间[a,b]上使用二分法寻找f(x)=0的根"""
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        return (a + b) / 2 if abs(fa) < abs(fb) else a
    
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        if abs(fc) < tol or abs(b - a) < tol:
            return c
        
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    
    return (a + b) / 2

# ===== 主计算函数 =====
def main():
    # 时间窗口定义
    t_start = t_detonate
    t_end = t_detonate + t_window
    
    # 时间采样点
    num_points = 2000
    ts = np.linspace(t_start, t_end, num_points)
    
    # 定义距离函数
    def f_threshold(t):
        return los_distance_sq_to_cloud_center(t) - r_cloud**2
    
    # 计算各时间点的距离平方
    d2_list = np.array([los_distance_sq_to_cloud_center(t) for t in ts], dtype=float)
    covered = d2_list <= (r_cloud**2 + 1e-12)  # 容差判断
    
    # 寻找遮蔽区间的边界
    boundaries = []
    if covered.any():
        for i in range(len(ts) - 1):
            if covered[i] != covered[i+1]:
                root = bisection_root(f_threshold, ts[i], ts[i+1], tol=1e-10, max_iter=100)
                boundaries.append(root)
        
        # 补齐端点
        if covered[0]:
            boundaries = [t_start] + boundaries
        if covered[-1]:
            boundaries = boundaries + [t_end]
        
        # 转换为成对区间
        intervals = [(boundaries[i], boundaries[i+1]) for i in range(0, len(boundaries)-1, 2)]
        total_covered = float(sum(b - a for a, b in intervals))
    else:
        intervals = []
        total_covered = 0.0
    
    # ===== 结果输出 =====
    print("=== 关键参数 ===")
    print(f"起爆时刻 t_det = {t_detonate:.3f} s")
    print(f"起爆点 C_det = {bomb_position(t_detonate)}")
    print(f"有效时窗: [{t_start:.3f}, {t_end:.3f}] s")
    
    print("\n=== 遮蔽区间 ===")
    for i, (start, end) in enumerate(intervals):
        print(f"区间 {i+1}: {start:.6f} s ~ {end:.6f} s, 时长: {end-start:.6f} s")
    
    print(f"\n总有效遮蔽时长: {total_covered:.6f} s")
    
    # ===== 可视化 =====
    # 3D几何示意图
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 计算关键点轨迹
    missile_traj = np.array([missile_position(t) for t in np.linspace(0, 30, 100)])
    drone_traj = np.array([drone_position(t) for t in np.linspace(0, 30, 100)])
    bomb_traj = np.array([bomb_position(t) for t in np.linspace(t_release, t_detonate, 50)])
    cloud_traj = np.array([cloud_center_position(t) for t in np.linspace(t_detonate, t_end, 50)])
    
    # 绘制轨迹
    ax.plot(missile_traj[:,0], missile_traj[:,1], missile_traj[:,2], 
            'r-', linewidth=1.5, label='导弹轨迹')
    ax.plot(drone_traj[:,0], drone_traj[:,1], drone_traj[:,2], 
            'b-', linewidth=1.5, label='无人机轨迹')
    ax.plot(bomb_traj[:,0], bomb_traj[:,1], bomb_traj[:,2], 
            'g-', linewidth=1.5, label='干扰弹轨迹')
    ax.plot(cloud_traj[:,0], cloud_traj[:,1], cloud_traj[:,2], 
            'm-', linewidth=1.5, label='云团中心轨迹')
    
    # 标记关键点
    ax.scatter(*M0, c='red', s=60, label='导弹初始位置')
    ax.scatter(*F0, c='blue', s=60, label='无人机初始位置')
    ax.scatter(*bomb_position(t_release), c='green', s=60, label='干扰弹投放点')
    ax.scatter(*bomb_position(t_detonate), c='magenta', s=80, marker='*', label='起爆点')
    ax.scatter(*T, c='black', s=100, marker='X', label='真目标')
    
    # 设置图形属性
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('烟幕干扰弹投放策略几何示意图')
    ax.legend()
    
    # 距离-时间曲线
    fig2 = plt.figure(figsize=(9, 5))
    ax2 = fig2.add_subplot(111)
    d_arr = np.sqrt(np.clip(d2_list, 0, np.inf))
    mask_covered = (d_arr <= r_cloud) & np.isfinite(d_arr)
    
    ax2.plot(ts, d_arr, linewidth=1.5, label="视线-云团中心最小距离 d(t)")
    ax2.axhline(r_cloud, linestyle="--", linewidth=1.0, color='red', label="阈值 r=10 m")
    
    # 填充遮蔽区间
    ax2.fill_between(ts, 0, d_arr, where=mask_covered, 
                    alpha=0.2, color='green', label="遮蔽区间")
    
    ax2.set_xlabel("时间 (s)")
    ax2.set_ylabel("距离 (m)")
    ax2.set_title("问题一: 视线-云团中心距离随时间的变化")
    ax2.grid(True, linestyle=":")
    ax2.legend(loc="best")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()