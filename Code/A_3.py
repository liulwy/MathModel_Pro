import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import json
import math


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 常量定义
T = np.array([0.0, 200.0, 0.0])  # 真目标位置
M0 = np.array([20000.0, 0.0, 2000.0])  # 导弹初始位置
F0 = np.array([17800.0, 0.0, 1800.0])  # 无人机初始位置
vM = 300.0  # 导弹速度
g = 9.8  # 重力加速度
v_sink = 3.0  # 云团下沉速度
r_cloud = 10.0  # 云团有效半径
t_window = 20.0  # 云团有效时间
N = 10  # 目标轴线采样点数
DT = 0.02  # 时间步长

# 导弹方向向量
uM = -M0 / np.linalg.norm(M0)

# 使用问题二优化结果
theta_deg = 9.090  # 最优航向角(度)
vF = 71.162  # 最优速度(m/s)
theta_star = np.radians(theta_deg)  # 转换为弧度
uF = np.array([np.cos(theta_star), np.sin(theta_star), 0])  # 无人机方向向量

def missile_position(t):
    return M0 + vM * uM * t

def drone_position(t):
    return F0 + vF * t * uF

def bomb_position(t, t_rel):
    if t < t_rel:
        return drone_position(t)
    
    dt = t - t_rel
    rel_pos = drone_position(t_rel)
    return rel_pos + vF * uF * dt + np.array([0, 0, -0.5 * g * dt**2])

def cloud_center_position(t, t_det):
    det_pos = bomb_position(t_det, t_det)  # 起爆点位置
    if t < t_det:
        return np.array([np.nan, np.nan, np.nan])
    
    dt = t - t_det
    return det_pos + np.array([0, 0, -v_sink * dt])

# 计算点到线段的最小距离平方
def min_distance_sq_to_line(A, B, P):
    AB = B - A
    AP = P - A
    AB_sq = np.dot(AB, AB)
    
    if AB_sq < 1e-12:
        return np.dot(AP, AP)
    
    s = np.dot(AP, AB) / AB_sq
    s_clamped = max(0.0, min(1.0, s))
    Q = A + s_clamped * AB
    return np.dot(P - Q, P - Q)

# 计算三枚弹的最小距离平方
def min_d2(t, params):
    d2_list = []
    M_t = missile_position(t)
    
    for j in range(3):
        t_rel_j = params[2*j]
        t_det_j = t_rel_j + params[2*j+1]
        C_t = cloud_center_position(t, t_det_j)
        
        if any(np.isnan(C_t)):
            d2_list.append(np.inf)
            continue
        
        # 计算目标轴线上的N个采样点
        d2_samples = []
        for alpha in np.linspace(0, 1, N):
            T_i = T + np.array([0, 0, 10*alpha])  # 目标轴线采样点
            d2 = min_distance_sq_to_line(T_i, M_t, C_t)
            d2_samples.append(d2)
        
        d2_list.append(min(d2_samples))
    
    return min(d2_list)

# 遮蔽时长计算函数
def covered_time(params):
    t_det_list = [params[2*j] + params[2*j+1] for j in range(3)]
    t_start = min(t_det_list)
    t_end = max(t_det_list) + t_window
    
    ts = np.arange(t_start, t_end, DT)
    covered = np.zeros(len(ts), dtype=bool)
    
    for i, t in enumerate(ts):
        d2_min = min_d2(t, params)
        covered[i] = d2_min <= (r_cloud**2 + 1e-12)
    
    return np.sum(covered) * DT

# 局部优化函数
def refine(params, iters=240, step=(0.5, 0.5)):
    cur = (covered_time(params), params.copy())
    step_sizes = np.array([step[0]]*3 + [step[1]]*3)
    
    for k in range(iters):
        new_params = params + np.random.uniform(-step_sizes, step_sizes)
        
        # 确保时间约束：t_rel1 < t_rel2 < t_rel3 且间隔≥1s
        new_params[0] = max(0, new_params[0])
        new_params[2] = max(new_params[0] + 1, new_params[2])
        new_params[4] = max(new_params[2] + 1, new_params[4])
        
        # 确保延迟时间≥0
        new_params[1::2] = np.maximum(0, new_params[1::2])
        
        # 计算新解
        val = covered_time(new_params)
        
        # 接受更优解
        if val > cur[0]:
            cur = (val, new_params)
        
        # 每40次迭代缩小步长
        if (k+1) % 40 == 0:
            step_sizes *= 0.6
    
    return cur

# 主优化流程
def main():
    # 粗网格搜索（简化示例）
    best_score = -np.inf
    best_params = None
    
    # 参数范围示例（实际应更全面）
    t_rel1_values = np.linspace(0, 5, 6)
    delta_t1_values = np.linspace(0, 3, 4)
    
    for t_rel1 in t_rel1_values:
        for delta_t1 in delta_t1_values:
            # 固定后两枚弹的投放时间间隔
            t_rel2 = t_rel1 + 1.5
            t_rel3 = t_rel2 + 2.0
            delta_t2 = 1.0
            delta_t3 = 2.0
            
            params = np.array([t_rel1, delta_t1, t_rel2, delta_t2, t_rel3, delta_t3])
            score = covered_time(params)
            
            if score > best_score:
                best_score = score
                best_params = params
    
    # 局部细化
    best_score, best_params = refine(best_params, iters=240)
    
    # 提取优化结果
    t_rel = [best_params[0], best_params[2], best_params[4]]
    delta_t = [best_params[1], best_params[3], best_params[5]]
    t_det = [t_rel[0] + delta_t[0], t_rel[1] + delta_t[1], t_rel[2] + delta_t[2]]
    
    # 计算精确遮蔽区间
    t_min = min(t_det)
    t_max = max(t_det) + t_window
    ts = np.linspace(t_min, t_max, 2000)
    d_min = np.array([min_d2(t, best_params) for t in ts])
    covered = d_min <= (r_cloud**2 + 1e-12)
    
    # 结果输出
    print(f"最优投放时刻: {t_rel}")
    print(f"最优起爆延迟: {delta_t}")
    print(f"起爆时刻: {t_det}")
    print(f"总遮蔽时长: {best_score:.6f} s")
    
    # 保存结果
    result = {
        "theta_deg": theta_deg,
        "vF": vF,
        "t_rel": t_rel,
        "delta_t": delta_t,
        "t_det": t_det,
        "total_time": best_score
    }
    
    with open("result1.json", "w") as f:
        json.dump(result, f, indent=2)
    
    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ts, np.sqrt(d_min), label="d_min(t)")
    ax.axhline(r_cloud, linestyle="--", color="r", label="阈值 r=10 m")
    ax.fill_between(ts, 0, np.sqrt(d_min), where=covered, alpha=0.2, label="遮蔽区间")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("距离 (m)")
    ax.set_title("三弹协同策略下的最小距离曲线")
    ax.legend()
    plt.savefig("distance_curve.png", dpi=300)
    
    # 3D轨迹可视化
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    
    # 导弹轨迹
    t_missile = np.linspace(0, 30, 100)
    M_traj = np.array([missile_position(t) for t in t_missile])
    ax.plot(M_traj[:, 0], M_traj[:, 1], M_traj[:, 2], "r-", label="导弹轨迹")
    
    # 无人机轨迹
    t_drone = np.linspace(0, max(t_rel)+5, 100)
    F_traj = np.array([drone_position(t) for t in t_drone])
    ax.plot(F_traj[:, 0], F_traj[:, 1], F_traj[:, 2], "b-", label="无人机轨迹")
    
    # 云团轨迹
    colors = ["g", "m", "c"]
    for j in range(3):
        t_cloud = np.linspace(t_det[j], t_det[j] + t_window, 50)
        C_traj = np.array([cloud_center_position(t, t_det[j]) for t in t_cloud])
        ax.plot(C_traj[:, 0], C_traj[:, 1], C_traj[:, 2], f"{colors[j]}-", label=f"云团{j+1}轨迹")
    
    # 标记关键点
    ax.scatter(*M0, c="red", s=50, label="导弹初始位置")
    ax.scatter(*F0, c="blue", s=50, label="无人机初始位置")
    ax.scatter(*T, c="black", s=80, marker="*", label="真目标")
    
    # 设置图形属性
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("三弹协同策略几何示意图")
    ax.legend()
    plt.savefig("trajectories_3d.png", dpi=300)

if __name__ == "__main__":
    main()