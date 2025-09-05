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
# 真目标圆柱体：半径7米，高度10米，中心轴从(0,200,0)到(0,200,10)
T_center = np.array([0.0, 200.0, 5.0], dtype=float)  # 圆柱体中心点
T_radius = 7.0  # 圆柱体半径(m)
T_height = 10.0  # 圆柱体高度(m)

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
def point_to_line_segment_distance_sq(point, line_start, line_end):
    """计算点到线段的最小距离平方"""
    # 线段向量
    line_vec = line_end - line_start
    # 点到线段起点的向量
    point_vec = point - line_start
    # 线段长度的平方
    line_len_sq = np.dot(line_vec, line_vec)
    
    # 如果线段长度为零，则直接计算点到点的距离
    if line_len_sq < 1e-10:
        return np.dot(point_vec, point_vec)
    
    # 计算投影比例
    t = np.dot(point_vec, line_vec) / line_len_sq
    t = max(0.0, min(1.0, t))  # 钳制到[0,1]区间
    
    # 计算投影点
    projection = line_start + t * line_vec
    
    # 返回点到投影点的距离平方
    return np.sum((point - projection)**2)

def is_point_covered(t: float, target_point: np.ndarray) -> bool:
    """检查特定目标点是否被遮蔽"""
    cloud_center = cloud_center_position(t)
    if any(np.isnan(cloud_center)):
        return False
    
    # 导弹位置
    M_t = missile_position(t)
    
    # 计算云团中心到导弹-目标点线段的最小距离平方
    d2 = point_to_line_segment_distance_sq(cloud_center, M_t, target_point)
    
    # 检查是否在云团有效半径内
    return d2 <= (r_cloud**2 + 1e-12)

def generate_cylinder_points(num_points=100):
    """在圆柱体表面生成均匀分布的点（不包括底面）"""
    points = []
    
    # 圆柱体参数
    z_min = T_center[2] - T_height/2
    z_max = T_center[2] + T_height/2
    
    # 计算侧面和顶面的面积比例
    area_side = 2 * np.pi * T_radius * T_height
    area_top = np.pi * T_radius**2
    total_area = area_side + area_top
    
    # 分配点数
    num_side = int(num_points * area_side / total_area)
    num_top = num_points - num_side
    
    # 生成侧面点
    num_z = max(2, int(np.sqrt(num_side / (2 * np.pi))))
    num_theta = max(4, int(num_side / num_z))
    
    z_values = np.linspace(z_min, z_max, num_z)
    theta_values = np.linspace(0, 2*np.pi, num_theta, endpoint=False)
    
    for z in z_values:
        for theta in theta_values:
            x = T_center[0] + T_radius * np.cos(theta)
            y = T_center[1] + T_radius * np.sin(theta)
            points.append(np.array([x, y, z]))
    
    # 生成顶面点
    num_r = max(1, int(np.sqrt(num_top)))
    num_theta_top = max(4, int(num_top / num_r))
    
    r_values = np.linspace(0, T_radius, num_r)
    theta_values_top = np.linspace(0, 2*np.pi, num_theta_top, endpoint=False)
    
    # 顶面
    for r in r_values:
        for theta in theta_values_top:
            x = T_center[0] + r * np.cos(theta)
            y = T_center[1] + r * np.sin(theta)
            points.append(np.array([x, y, z_max]))
    
    return points[:num_points]  # 确保返回指定数量的点

# 生成圆柱体表面点（不包括底面）
cylinder_points = generate_cylinder_points(100)

def los_distance_sq_each_ray(t: float) -> np.ndarray:
    """返回导弹到圆柱体采样点集合的视线到云团中心的最小距离平方数组"""
    C_t = cloud_center_position(t)
    if np.isnan(C_t).any():
        return np.full(len(cylinder_points), np.inf, dtype=float)

    M_t = missile_position(t)
    T_arr = np.asarray(cylinder_points, dtype=float)  # (N,3)

    # 线段最近点向量化计算：对每个 T_i -- M_t
    TM = M_t - T_arr                        # (N,3)
    den = np.einsum('ij,ij->i', TM, TM)     # (N,)
    CT = C_t - T_arr                        # (N,3)

    s = np.zeros_like(den, dtype=float)
    mask = den > 0.0
    s[mask] = np.einsum('ij,ij->i', CT[mask], TM[mask]) / den[mask]
    s = np.clip(s, 0.0, 1.0)

    closest = T_arr + s[:, None] * TM       # (N,3)
    diff = closest - C_t
    d2 = np.einsum('ij,ij->i', diff, diff)  # (N,)
    return d2

def is_target_covered(t: float) -> bool:
    """全覆盖判定：所有射线的最短距离均不超过 r_cloud 才算有效"""
    d2_each = los_distance_sq_each_ray(t)
    return bool(np.all(d2_each <= (r_cloud**2 + 1e-12)))

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
    num_points = 1000  # 减少采样点以提高性能
    ts = np.linspace(t_start, t_end, num_points)
    
    # 定义遮蔽函数
    def f_threshold(t):
        return 1.0 if is_target_covered(t) else -1.0
    
    # 计算各时间点的遮蔽状态
    covered = np.array([is_target_covered(t) for t in ts], dtype=bool)
    
    # 寻找遮蔽区间的边界
    boundaries = []
    if covered.any():
        for i in range(len(ts) - 1):
            if covered[i] != covered[i+1]:
                # 使用二分法精确查找边界
                root = bisection_root(lambda t: 1.0 if is_target_covered(t) else -1.0, 
                                     ts[i], ts[i+1], tol=1e-10, max_iter=100)
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
    fig = plt.figure(figsize=(14, 10))
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
    
    # 绘制圆柱体目标（不包括底面）
    z_min = T_center[2] - T_height/2
    z_max = T_center[2] + T_height/2
    
    # 绘制侧面
    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(z_min, z_max, 10)
    theta, z = np.meshgrid(theta, z)
    x = T_center[0] + T_radius * np.cos(theta)
    y = T_center[1] + T_radius * np.sin(theta)
    ax.plot_surface(x, y, z, color='gray', alpha=0.3, label='圆柱体目标')
    
    # 绘制顶面
    r = np.linspace(0, T_radius, 10)
    theta = np.linspace(0, 2 * np.pi, 100)
    r, theta = np.meshgrid(r, theta)
    x = T_center[0] + r * np.cos(theta)
    y = T_center[1] + r * np.sin(theta)
    z_top = np.full_like(x, z_max)
    ax.plot_surface(x, y, z_top, color='gray', alpha=0.3)
    
    # 绘制采样点（不包括底面）
    points_array = np.array(cylinder_points)
    ax.scatter(points_array[:,0], points_array[:,1], points_array[:,2], 
               c='black', s=10, alpha=0.5, label='采样点')
    
    # 标记关键点
    ax.scatter(*M0, c='red', s=60, label='导弹初始位置')
    ax.scatter(*F0, c='blue', s=60, label='无人机初始位置')
    ax.scatter(*bomb_position(t_release), c='green', s=60, label='干扰弹投放点')
    ax.scatter(*bomb_position(t_detonate), c='magenta', s=100, marker='*', label='起爆点')
    
    # 添加文字标注
    ax.text(T_center[0], T_center[1], T_center[2], "圆柱体目标中心", fontsize=12, color='black', zorder=10)
    
    # 设置图形属性
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('烟幕干扰弹投放策略几何示意图', fontsize=14)
    ax.legend(fontsize=10)
    
    # 调整视角以便更好地观察圆柱体目标
    ax.view_init(elev=25, azim=-45)  # 仰角25度，方位角-45度
    
    # 设置坐标轴范围
    ax.set_xlim([-1000, 20000])
    ax.set_ylim([-1000, 1000])
    ax.set_zlim([-100, 2500])
    
    # 添加网格
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # 距离-时间曲线（改为遮蔽状态图）
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)
    
    # 计算遮蔽状态
    covered_status = np.array([is_target_covered(t) for t in ts], dtype=float)
    
    # 绘制遮蔽状态
    ax2.plot(ts, covered_status, 'b-', linewidth=1.5, label="遮蔽状态")
    ax2.axhline(1.0, linestyle="--", linewidth=1.0, color='green', label="完全遮蔽")
    ax2.axhline(0.0, linestyle="--", linewidth=1.0, color='red', label="未完全遮蔽")
    
    # 填充遮蔽区间
    mask_covered = covered_status > 0.5
    ax2.fill_between(ts, 0, covered_status, where=mask_covered, 
                    alpha=0.2, color='green', label="完全遮蔽区间")
    
    # 标记遮蔽区间
    for i, (start, end) in enumerate(intervals):
        ax2.axvspan(start, end, alpha=0.1, color='blue')
        ax2.text((start+end)/2, 0.5, f"区间{i+1}", ha='center', fontsize=10)
    
    ax2.set_xlabel("时间 (s)", fontsize=12)
    ax2.set_ylabel("遮蔽状态", fontsize=12)
    ax2.set_title("问题一: 圆柱体目标遮蔽状态随时间变化", fontsize=14)
    ax2.grid(True, linestyle=":")
    ax2.legend(loc="best", fontsize=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()