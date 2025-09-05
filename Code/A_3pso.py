# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import multiprocessing as mp  # 添加多进程支持
from functools import partial

# 字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===== 常量定义（与前两问一致）=====
# 圆柱体目标
T_center = np.array([0.0, 200.0, 5.0], dtype=float)  # 圆柱体中心点
T_radius = 7.0                                       # 半径(m)
T_height = 10.0                                      # 高度(m)

# 导弹与无人机
M0 = np.array([20000.0, 0.0, 2000.0], dtype=float)   # 导弹初始位置
vM = 300.0                                           # 导弹速度(m/s)
uM = -M0 / np.linalg.norm(M0)                        # 导弹方向单位向量
F0 = np.array([17800.0, 0.0, 1800.0], dtype=float)   # 无人机初始位置

# 物理参数
g = 9.8                   # 重力加速度(m/s^2)
v_sink = 3.0              # 云团下沉速度(m/s)
r_cloud = 10.0            # 云团有效半径(m)
t_window = 20.0           # 云团有效时间(s)
DT = 0.02                 # 时间步长(s)

# ===== 圆柱体表面采样（侧面+顶面，不含底面）=====
def generate_cylinder_points(num_points=150):
    """在圆柱体表面生成近似等面积的采样点（不含底面）"""
    points = []

    z_min = T_center[2] - T_height / 2.0
    z_max = T_center[2] + T_height / 2.0

    area_side = 2 * np.pi * T_radius * T_height
    area_top = np.pi * T_radius**2
    total_area = area_side + area_top

    num_side = int(num_points * area_side / total_area)
    num_top = max(0, num_points - num_side)

    # 侧面：z-θ 规则网格（近似等面积）
    num_z = max(3, int(np.sqrt(max(num_side, 1) / (2 * np.pi))))
    num_theta = max(6, int(max(num_side, 1) / max(num_z, 1)))
    z_vals = np.linspace(z_min, z_max, num_z)
    th_vals = np.linspace(0, 2*np.pi, num_theta, endpoint=False)
    for z in z_vals:
        for th in th_vals:
            x = T_center[0] + T_radius * np.cos(th)
            y = T_center[1] + T_radius * np.sin(th)
            points.append([x, y, z])

    # 顶面：极坐标网格（简化近似）
    num_r = max(2, int(np.sqrt(max(num_top, 1))))
    num_th_top = max(6, int(max(num_top, 1) / max(num_r, 1)))
    r_vals = np.linspace(0, T_radius, num_r)
    th_top = np.linspace(0, 2*np.pi, num_th_top, endpoint=False)
    for r in r_vals:
        for th in th_top:
            x = T_center[0] + r * np.cos(th)
            y = T_center[1] + r * np.sin(th)
            points.append([x, y, z_max])

    pts = np.asarray(points[:num_points], dtype=float)
    return pts

T_arr = generate_cylinder_points(150)  # (N,3)，可按需加密

# ===== 运动学 =====
def missile_position(t: float) -> np.ndarray:
    return M0 + vM * t * uM

def drone_position(t: float, vF: float, theta: float) -> np.ndarray:
    uF = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)
    return F0 + vF * t * uF

def bomb_position(t: float, t_rel: float, vF: float, theta: float) -> np.ndarray:
    """t < t_rel 返回无人机位置；t >= t_rel 抛体"""
    uF = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)
    if t < t_rel:
        return F0 + vF * t * uF
    dt = t - t_rel
    rel_pos = F0 + vF * t_rel * uF
    return rel_pos + vF * uF * dt + np.array([0.0, 0.0, -0.5 * g * dt**2])

def cloud_center_position(t: float, t_det: float, vF: float, theta: float) -> np.ndarray:
    """t_det = t_rel + dly； t < t_det 返回 NaN"""
    if t < t_det:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    C_det = bomb_position(t_det, t_det, vF, theta)  # 起爆点
    dt = t - t_det
    return C_det + np.array([0.0, 0.0, -v_sink * dt], dtype=float)

# ===== 距离与遮蔽判据 =====
_thr2 = r_cloud**2 + 1e-12

# 添加角度规范化函数
def angle_wrap(theta):
    """将角度规范化到 [-π, π] 范围内"""
    return (theta + np.pi) % (2 * np.pi) - np.pi

def los_distance_sq_each_ray(M_t: np.ndarray, C_t: np.ndarray) -> np.ndarray:
    """返回 M_t->T_arr 每条线段到 C_t 的最小距离平方数组"""
    TM = M_t - T_arr                    # (N,3)
    den = np.einsum('ij,ij->i', TM, TM) # (N,)
    CT = C_t - T_arr                    # (N,3)

    s = np.zeros_like(den, dtype=float)
    mask = den > 0.0
    s[mask] = np.einsum('ij,ij->i', CT[mask], TM[mask]) / den[mask]
    s = np.clip(s, 0.0, 1.0)

    closest = T_arr + s[:, None] * TM   # (N,3)
    diff = closest - C_t
    d2 = np.einsum('ij,ij->i', diff, diff)
    return d2

def covered_time_3bombs(x: np.ndarray) -> float:
    """
    8维决策变量:
      x = [theta, vF, t1, d1, gap2, d2, gap3, d3]
      t2 = t1 + 1 + max(0, gap2), t3 = t2 + 1 + max(0, gap3)
      d_k >= 0
    遮蔽判据：存在任意一枚云团使得所有视线均在 r_cloud 内，则视为遮蔽成立（联合“或”）。
    """
    theta = np.clip(x[0], -np.pi, np.pi)
    vF = np.clip(x[1], 70.0, 140.0)
    t1 = np.clip(x[2], 0.0, 10.0)
    d1 = np.clip(x[3], 0.0, 6.0)
    gap2 = max(0.0, np.clip(x[4], 0.0, 6.0))
    d2 = np.clip(x[5], 0.0, 6.0)
    gap3 = max(0.0, np.clip(x[6], 0.0, 6.0))
    d3 = np.clip(x[7], 0.0, 6.0)

    t2 = t1 + 1.0 + gap2
    t3 = t2 + 1.0 + gap3
    dets = np.array([t1 + d1, t2 + d2, t3 + d3], dtype=float)

    t_start = float(np.min(dets))
    t_end = float(np.max(dets) + t_window)
    if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
        return 0.0

    ts = np.arange(t_start, t_end, DT)
    total = 0.0

    for t in ts:
        Mt = missile_position(t)
        # 三枚云中心（允许未起爆 NaN）
        C1 = cloud_center_position(t, t1 + d1, vF, theta)
        C2 = cloud_center_position(t, t2 + d2, vF, theta)
        C3 = cloud_center_position(t, t3 + d3, vF, theta)

        covered = False
        for Ct in (C1, C2, C3):
            if np.isnan(Ct).any():
                continue
            d2_each = los_distance_sq_each_ray(Mt, Ct)
            if np.all(d2_each <= _thr2):
                covered = True
                break

        if covered:
            total += DT

    return float(total)

def covered_time_3bombs_fast(x: np.ndarray) -> float:
    """
    优化版三弹遮蔽时长计算 - 向量化+早期终止
    """
    theta = np.clip(x[0], -np.pi, np.pi)
    vF = np.clip(x[1], 70.0, 140.0)
    t1 = np.clip(x[2], 0.0, 10.0)
    d1 = np.clip(x[3], 0.0, 6.0)
    gap2 = max(0.0, np.clip(x[4], 0.0, 6.0))
    d2 = np.clip(x[5], 0.0, 6.0)
    gap3 = max(0.0, np.clip(x[6], 0.0, 6.0))
    d3 = np.clip(x[7], 0.0, 6.0)

    t2 = t1 + 1.0 + gap2
    t3 = t2 + 1.0 + gap3
    
    t_rels = np.array([t1, t2, t3])
    dlys = np.array([d1, d2, d3])
    dets = t_rels + dlys

    t_start = float(np.min(dets))
    t_end = float(np.max(dets) + t_window)
    if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
        return 0.0

    # 自适应采样 - 解析策略：先粗采样，再细化
    # 第一阶段：粗采样快速估计
    dt_coarse = 0.1  # 粗采样步长
    ts_coarse = np.arange(t_start, t_end, dt_coarse)
    
    # 预计算无人机方向向量(复用)
    uF = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)
    
    # 预计算起爆点(复用)
    C_dets = []
    for t_rel, dly in zip(t_rels, dlys):
        t_det = t_rel + dly
        # 投放点
        rel_pos = F0 + vF * t_rel * uF
        # 起爆点(考虑抛体运动)
        dt_to_det = dly
        det_pos = rel_pos + vF * uF * dt_to_det + np.array([0.0, 0.0, -0.5 * g * dt_to_det**2])
        C_dets.append(det_pos)
    C_dets = np.array(C_dets)
    
    # 粗采样评估
    covered_coarse = np.zeros_like(ts_coarse, dtype=bool)
    
    for i, t in enumerate(ts_coarse):
        Mt = M0 + vM * t * uM  # 导弹位置
        
        # 检查每个云团
        for j, (t_det, C_det) in enumerate(zip(dets, C_dets)):
            # 跳过未起爆的云团
            if t < t_det:
                continue
                
            # 云团中心位置
            dt_from_det = t - t_det
            if dt_from_det > t_window:
                continue  # 超过有效时间
                
            Ct = C_det + np.array([0.0, 0.0, -v_sink * dt_from_det])
            
            # 检查遮蔽
            d2_each = los_distance_sq_each_ray(Mt, Ct)
            if np.all(d2_each <= _thr2):
                covered_coarse[i] = True
                break
    
    # 找到遮蔽区间(可能有多个)
    total_time = 0.0
    if np.any(covered_coarse):
        # 识别状态变化的位置
        changes = np.diff(covered_coarse.astype(int))
        start_idxs = np.where(changes == 1)[0]
        end_idxs = np.where(changes == -1)[0]
        
        # 处理边界情况
        if covered_coarse[0]:
            start_idxs = np.insert(start_idxs, 0, -1)
        if covered_coarse[-1]:
            end_idxs = np.append(end_idxs, len(covered_coarse)-1)
            
        # 细化每个区间边界
        for start_idx, end_idx in zip(start_idxs, end_idxs):
            t_a = ts_coarse[start_idx + 1] if start_idx >= 0 else ts_coarse[0]
            t_b = ts_coarse[end_idx + 1] if end_idx < len(ts_coarse) - 1 else ts_coarse[-1]
            
            # 在细化区间使用更精确的步长
            ts_fine = np.arange(t_a, t_b, DT)
            if len(ts_fine) == 0:
                continue
                
            covered_fine = np.zeros_like(ts_fine, dtype=bool)
            
            for i, t in enumerate(ts_fine):
                Mt = M0 + vM * t * uM
                
                for j, (t_det, C_det) in enumerate(zip(dets, C_dets)):
                    if t < t_det or t > t_det + t_window:
                        continue
                        
                    dt_from_det = t - t_det
                    Ct = C_det + np.array([0.0, 0.0, -v_sink * dt_from_det])
                    
                    d2_each = los_distance_sq_each_ray(Mt, Ct)
                    if np.all(d2_each <= _thr2):
                        covered_fine[i] = True
                        break
            
            # 计算这个区间的遮蔽时长
            interval_time = np.sum(covered_fine) * DT
            total_time += interval_time
    
    return float(total_time)

def _eval_particle(x):
    """粒子评估函数(用于并行处理)"""
    return covered_time_3bombs_fast(x)

class PSO:
    def __init__(self, swarm=60, iters=300, w0=0.9, w1=0.4, c1=1.8, c2=1.8, seed=42):
        self.rng = np.random.default_rng(seed)
        self.swarm = swarm
        self.iters = iters
        self.w0, self.w1 = w0, w1
        self.c1, self.c2 = c1, c2

        # 边界: [theta, vF, t1, d1, gap2, d2, gap3, d3]
        self.lb = np.array([-np.pi, 70.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.ub = np.array([ np.pi,140.0,10.0, 6.0, 6.0, 6.0, 6.0, 6.0], dtype=float)
        span = self.ub - self.lb

        self.X = self.lb + self.rng.random((swarm, 8)) * span
        self.X[:, 0] = np.array([angle_wrap(t) for t in self.X[:, 0]])  # theta wrap
        self.vmax = 0.2 * span
        self.V = self.rng.uniform(-self.vmax, self.vmax, size=(swarm, 8))

        self.fit = np.zeros(swarm, dtype=float)
        self.pbest = self.X.copy()
        self.pfit = np.full(swarm, -np.inf, dtype=float)
        self.gbest = None
        self.gfit = -np.inf

        # 可视化
        self.it_hist = []
        self.best_hist = []

    def eval_one(self, x):
        return covered_time_3bombs(x)

    def optimize(self):
        # 初始化评估
        self.fit = np.array([self.eval_one(x) for x in self.X], dtype=float)
        self.pbest = self.X.copy()
        self.pfit = self.fit.copy()
        idx = int(np.argmax(self.pfit))
        self.gbest = self.pbest[idx].copy()
        self.gfit = float(self.pfit[idx])

        # 记录收敛历史但不绘图
        self.it_hist = [0]
        self.best_hist = [self.gfit]
        print(f"[PSO 0/{self.iters}] iter_best={np.max(self.fit):.6f}s, gbest={self.gfit:.6f}s")

        for it in range(1, self.iters + 1):
            w = self.w0 + (self.w1 - self.w0) * (it / self.iters)
            r1 = self.rng.random((self.swarm, 8))
            r2 = self.rng.random((self.swarm, 8))

            # 更新速度与位置
            self.V = w * self.V + self.c1 * r1 * (self.pbest - self.X) + self.c2 * r2 * (self.gbest - self.X)
            self.V = np.clip(self.V, -self.vmax, self.vmax)
            self.X = self.X + self.V

            # 边界与角度
            self.X = np.clip(self.X, self.lb, self.ub)
            self.X[:, 0] = np.array([angle_wrap(t) for t in self.X[:, 0]])

            # 评估并更新
            self.fit = np.array([self.eval_one(x) for x in self.X], dtype=float)
            better = self.fit > self.pfit
            self.pbest[better] = self.X[better]
            self.pfit[better] = self.fit[better]

            idx = int(np.argmax(self.pfit))
            iter_best = float(np.max(self.fit))
            if self.pfit[idx] > self.gfit:
                self.gbest = self.pbest[idx].copy()
                self.gfit = float(self.pfit[idx])

            # 只记录历史，不绘图
            print(f"[PSO {it}/{self.iters}] iter_best={iter_best:.6f}s, gbest={self.gfit:.6f}s")
            self.it_hist.append(it)
            self.best_hist.append(self.gfit)

        # 在最后绘制完整收敛曲线
        self.plot_convergence()
        
        return self.gfit, self.gbest
    
    def plot_convergence(self):
        """最终绘制完整收敛曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("PSO收敛曲线：三弹联合遮蔽时间", fontsize=14)
        ax.set_xlabel("迭代次数", fontsize=12)
        ax.set_ylabel("遮蔽时间 (s)", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.7)
        
        # 绘制收敛曲线
        ax.plot(self.it_hist, self.best_hist, "-o", ms=4, color="#FF7F0E", 
                linewidth=1.8, label="全局最优值")
        
        # 在最终点标记
        ax.scatter([self.it_hist[-1]], [self.best_hist[-1]], s=100, 
                   marker="*", color="#C44E52", label=f"最终值: {self.best_hist[-1]:.4f}s")
        
        # 添加标注
        for i in range(0, len(self.it_hist), max(1, len(self.it_hist)//10)):
            if i > 0 and self.best_hist[i] > self.best_hist[i-1]:
                ax.annotate(f"{self.best_hist[i]:.2f}", 
                           (self.it_hist[i], self.best_hist[i]),
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=9,
                           arrowprops=dict(arrowstyle='->', color='gray'))
        
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("pso_convergence_plot.png", dpi=300, bbox_inches='tight')
        plt.show()

class AcceleratedPSO:
    def __init__(self, swarm=60, iters=300, w0=0.9, w1=0.4, c1=1.8, c2=1.8, seed=42, 
                 n_jobs=-1, local_search_freq=10):
        self.rng = np.random.default_rng(seed)
        self.swarm = swarm
        self.iters = iters
        self.w0, self.w1 = w0, w1
        self.c1, self.c2 = c1, c2
        self.n_jobs = n_jobs if n_jobs > 0 else max(1, mp.cpu_count()-1)
        self.local_search_freq = local_search_freq

        # 边界: [theta, vF, t1, d1, gap2, d2, gap3, d3]
        self.lb = np.array([-np.pi, 70.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.ub = np.array([ np.pi,140.0,10.0, 6.0, 6.0, 6.0, 6.0, 6.0], dtype=float)
        span = self.ub - self.lb

        # 启发式初始化：包含一些基于问题结构的预设初始解
        self.X = np.zeros((swarm, 8), dtype=float)
        
        # 20%粒子使用均匀随机初始化
        num_random = int(swarm * 0.2)
        self.X[:num_random] = self.lb + self.rng.random((num_random, 8)) * span
        
        # 20%粒子围绕问题二的最优解扰动(假设某个角度最佳)
        num_p2 = int(swarm * 0.2)
        if num_p2 > 0:
            # 问题二的最佳解附近
            p2_best = np.array([np.radians(9.09), 71.16, 0.7, 3.5, 0, 0, 0, 0])
            self.X[num_random:num_random+num_p2] = p2_best + \
                self.rng.normal(0, 0.1, (num_p2, 8)) * span
        
        # 30%粒子尝试均匀分布的时序
        num_seq = int(swarm * 0.3)
        if num_seq > 0:
            start_idx = num_random + num_p2
            end_idx = start_idx + num_seq
            # 随机方向和速度
            self.X[start_idx:end_idx, 0] = self.rng.uniform(-np.pi, np.pi, num_seq)
            self.X[start_idx:end_idx, 1] = self.rng.uniform(70, 140, num_seq)
            # 均匀分布的时序
            for i in range(num_seq):
                # 随机第一发时间
                t1 = self.rng.uniform(0, 3)
                # 其余间隔尽量均匀
                self.X[start_idx+i, 2] = t1  # t1
                self.X[start_idx+i, 4] = self.rng.uniform(1, 3)  # gap2
                self.X[start_idx+i, 6] = self.rng.uniform(1, 3)  # gap3
                # 延迟也尝试均匀
                self.X[start_idx+i, 3] = self.rng.uniform(1, 4)  # d1
                self.X[start_idx+i, 5] = self.rng.uniform(1, 4)  # d2
                self.X[start_idx+i, 7] = self.rng.uniform(1, 4)  # d3
        
        # 剩余粒子随机初始化
        num_remaining = swarm - (num_random + num_p2 + num_seq)
        if num_remaining > 0:
            self.X[-(num_remaining):] = self.lb + self.rng.random((num_remaining, 8)) * span
        
        # 包装角度
        self.X[:, 0] = np.array([angle_wrap(t) for t in self.X[:, 0]])
        
        # 速度初始化
        self.vmax = 0.2 * span
        self.V = self.rng.uniform(-self.vmax, self.vmax, size=(swarm, 8))

        # 适应度
        self.fit = np.zeros(swarm, dtype=float)
        self.pbest = self.X.copy()
        self.pfit = np.full(swarm, -np.inf, dtype=float)
        self.gbest = None
        self.gfit = -np.inf

        # 可视化
        self.it_hist = []
        self.best_hist = []

    def local_search(self, x_center, radius=0.1):
        """在最优解附近进行局部搜索，提高精度"""
        best_x = x_center.copy()
        best_fit = covered_time_3bombs_fast(best_x)
        
        span = self.ub - self.lb
        n_trials = 20  # 局部搜索次数
        
        # 随机扰动搜索
        for _ in range(n_trials):
            x_new = x_center + self.rng.normal(0, radius, 8) * span
            x_new = np.clip(x_new, self.lb, self.ub)
            x_new[0] = angle_wrap(x_new[0])
            
            fit_new = covered_time_3bombs_fast(x_new)
            if fit_new > best_fit:
                best_fit = fit_new
                best_x = x_new.copy()
        
        return best_x, best_fit

    def optimize(self):
        # 创建进程池
        pool = mp.Pool(processes=self.n_jobs)
        eval_func = _eval_particle
        
        # 初始化评估 - 并行
        self.fit = np.array(pool.map(eval_func, self.X), dtype=float)
        self.pbest = self.X.copy()
        self.pfit = self.fit.copy()
        idx = int(np.argmax(self.pfit))
        self.gbest = self.pbest[idx].copy()
        self.gfit = float(self.pfit[idx])

        # 记录历史但不绘图
        self.it_hist = [0]
        self.best_hist = [self.gfit]
        print(f"[PSO 0/{self.iters}] iter_best={np.max(self.fit):.6f}s, gbest={self.gfit:.6f}s")

        for it in range(1, self.iters + 1):
            w = self.w0 + (self.w1 - self.w0) * (it / self.iters)
            r1 = self.rng.random((self.swarm, 8))
            r2 = self.rng.random((self.swarm, 8))

            # 更新速度与位置
            self.V = w * self.V + self.c1 * r1 * (self.pbest - self.X) + self.c2 * r2 * (self.gbest - self.X)
            self.V = np.clip(self.V, -self.vmax, self.vmax)
            self.X = self.X + self.V

            # 边界与角度
            self.X = np.clip(self.X, self.lb, self.ub)
            self.X[:, 0] = np.array([angle_wrap(t) for t in self.X[:, 0]])

            # 并行评估
            self.fit = np.array(pool.map(eval_func, self.X), dtype=float)
            
            # 更新个体最佳
            better = self.fit > self.pfit
            self.pbest[better] = self.X[better]
            self.pfit[better] = self.fit[better]

            idx = int(np.argmax(self.pfit))
            iter_best = float(np.max(self.fit))
            if self.pfit[idx] > self.gfit:
                self.gbest = self.pbest[idx].copy()
                self.gfit = float(self.pfit[idx])
            
            # 周期性局部搜索
            if it % self.local_search_freq == 0:
                improved_x, improved_fit = self.local_search(
                    self.gbest, 
                    radius=0.1 * (1 - it/self.iters)  # 随迭代递减搜索半径
                )
                if improved_fit > self.gfit:
                    self.gbest = improved_x.copy()
                    self.gfit = improved_fit
                    print(f"  [局部搜索] 发现更优解: {improved_fit:.6f}s")

            # 打印与曲线
            print(f"[PSO {it}/{self.iters}] iter_best={iter_best:.6f}s, gbest={self.gfit:.6f}s")
            self.it_hist.append(it)
            self.best_hist.append(self.gfit)

        # 最终局部精细搜索
        print("执行最终精细局部搜索...")
        improved_x, improved_fit = self.local_search(self.gbest, radius=0.02)
        if improved_fit > self.gfit:
            self.gbest = improved_x
            self.gfit = improved_fit
            print(f"  [精细搜索] 最终优化: {improved_fit:.6f}s")
            # 添加最后一个点
            self.it_hist.append(self.it_hist[-1] + 1)
            self.best_hist.append(self.gfit)
        
        # 关闭进程池
        pool.close()
        pool.join()
        
        # 最后绘制完整收敛曲线
        self.plot_convergence()
        
        return self.gfit, self.gbest
    
    def plot_convergence(self):
        """最终绘制完整收敛曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"加速PSO收敛曲线：三弹联合遮蔽时间 ({self.n_jobs}核并行)", fontsize=14)
        ax.set_xlabel("迭代次数", fontsize=12)
        ax.set_ylabel("遮蔽时间 (s)", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.7)
        
        # 绘制收敛曲线 
        ax.plot(self.it_hist, self.best_hist, "-o", ms=4, color="#FF7F0E", 
                linewidth=1.8, label="全局最优值")
        
        # 在最终点标记
        ax.scatter([self.it_hist[-1]], [self.best_hist[-1]], s=100, 
                   marker="*", color="#C44E52", label=f"最终值: {self.best_hist[-1]:.4f}s")
        
        # 添加标注
        improvements = np.diff(self.best_hist) > 0
        for i in range(1, len(self.it_hist)):
            if improvements[i-1] and i % max(1, len(self.it_hist)//10) == 0:
                ax.annotate(f"{self.best_hist[i]:.2f}", 
                           (self.it_hist[i], self.best_hist[i]),
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=9,
                           arrowprops=dict(arrowstyle='->', color='gray'))
        
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("pso_accelerated_convergence_plot.png", dpi=300, bbox_inches='tight')
        plt.show()

# ===== 主流程 =====
def decode_solution(x):
    theta = float(np.clip(x[0], -np.pi, np.pi))
    vF = float(np.clip(x[1], 70.0, 140.0))
    t1 = float(np.clip(x[2], 0.0, 10.0))
    d1 = float(np.clip(x[3], 0.0, 6.0))
    gap2 = float(max(0.0, np.clip(x[4], 0.0, 6.0)))
    d2 = float(np.clip(x[5], 0.0, 6.0))
    gap3 = float(max(0.0, np.clip(x[6], 0.0, 6.0)))
    d3 = float(np.clip(x[7], 0.0, 6.0))
    t2 = t1 + 1.0 + gap2
    t3 = t2 + 1.0 + gap3
    dets = [t1 + d1, t2 + d2, t3 + d3]
    return theta, vF, [t1, t2, t3], [d1, d2, d3], dets

def main():
    t0 = time.time()
    
    # 使用加速版PSO
    pso = AcceleratedPSO(
        swarm=60,             
        iters=200,            
        w0=0.9, w1=0.4,       
        c1=1.8, c2=1.8,       
        seed=42,              
        n_jobs=-1,             
        local_search_freq=10   
    )
    
    best_score, best_x = pso.optimize()  # 使用新的优化方法
    theta, vF, t_rel, dly, t_det = decode_solution(best_x)
    elapsed = time.time() - t0

    print("最优结果")
    print(f"theta: {np.degrees(theta):.4f}°")
    print(f"vF: {vF:.4f} m/s")
    print(f"t_rel: {t_rel}")
    print(f"dly:   {dly}")
    print(f"t_det: {t_det}")
    print(f"总遮蔽时长(联合): {best_score:.6f} s")
    print(f"用时: {elapsed:.2f} s")

    # 保存到 result1.xlsx（问题三要求）
    df = pd.DataFrame([{
        "theta_deg": np.degrees(theta),
        "vF": vF,
        "t_rel1": t_rel[0], "delay1": dly[0], "t_det1": t_det[0],
        "t_rel2": t_rel[1], "delay2": dly[1], "t_det2": t_det[1],
        "t_rel3": t_rel[2], "delay3": dly[2], "t_det3": t_det[2],
        "total_time": best_score
    }])
    df.to_excel("result1.xlsx", index=False)

if __name__ == "__main__":
    main()