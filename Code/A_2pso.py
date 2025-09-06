import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp  # 添加多进程支持
import signal
import functools
import time
import threading

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===== 常量定义 =====
fake_target = np.array([0.0, 0.0, 0.0])  # 假目标位置
T_center = np.array([0.0, 200.0, 5.0])    # 真目标圆柱体中心
T_radius = 7.0    # 圆柱体半径(m)
T_height = 10.0   # 圆柱体高度(m)

# 导弹参数
M0 = np.array([20000.0, 0.0, 2000.0])  # M1初始位置
vM = 300.0  # 导弹速度(m/s)
uM = (fake_target - M0) / np.linalg.norm(fake_target - M0)

# 无人机参数
F0 = np.array([17800.0, 0.0, 1800.0])  # FY1初始位置
min_speed, max_speed = 70, 140  # 速度范围(m/s)

# 物理参数
g = 9.8       # 重力加速度(m/s²)
v_sink = 3.0  # 云团下沉速度(m/s)
r_cloud = 10.0  # 云团有效半径(m)
t_window = 20.0  # 云团有效时间(s)
DT = 0.01     # 时间步长(s)

# ===== 圆柱体表面点生成 =====
def generate_cylinder_points(num_points=100):
    """生成圆柱体表面采样点（向量化实现）"""
    points = []
    z_min = T_center[2] - T_height/2
    z_max = T_center[2] + T_height/2
    
    # 侧面点（按面积比例分配点数）
    area_side = 2 * np.pi * T_radius * T_height
    area_top = np.pi * T_radius**2
    total_area = area_side + area_top
    num_side = int(num_points * area_side / total_area)
    num_top = num_points - num_side
    
    # 侧面采样
    num_theta = int(np.sqrt(num_side * T_radius / T_height))
    num_z = max(1, num_side // num_theta)
    
    theta = np.linspace(0, 2*np.pi, num_theta, endpoint=False)
    z = np.linspace(z_min, z_max, num_z)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    x = T_center[0] + T_radius * np.cos(theta_grid).flatten()
    y = T_center[1] + T_radius * np.sin(theta_grid).flatten()
    z = z_grid.flatten()
    
    side_points = np.column_stack((x, y, z))
    points.extend(side_points)
    
    # 顶面采样
    num_r = max(1, int(np.sqrt(num_top) / 2))
    num_theta_top = max(1, num_top // num_r)
    
    r = np.linspace(0, T_radius, num_r)
    theta_top = np.linspace(0, 2*np.pi, num_theta_top, endpoint=False)
    r_grid, theta_top_grid = np.meshgrid(r, theta_top)
    
    x_top = T_center[0] + r_grid.flatten() * np.cos(theta_top_grid.flatten())
    y_top = T_center[1] + r_grid.flatten() * np.sin(theta_top_grid.flatten())
    z_top = np.full_like(x_top, z_max)
    
    top_points = np.column_stack((x_top, y_top, z_top))
    points.extend(top_points)
    
    return points[:num_points]

cylinder_points = np.array(generate_cylinder_points(150))  # 增加采样点以提高精度

# ===== 向量化几何计算函数 =====
def point_to_line_segment_distance_sq_batch(point, line_start, line_end):
    """计算点到线段的最短距离平方（修复版）
    point: 云团中心点 (3,)
    line_start: 导弹位置 (3,)
    line_end: 目标点数组 (N,3)
    """
    # 计算线向量
    line_vec = line_end - line_start  # (N,3)
    
    # 计算从线段起点到目标点的向量
    point_vec = point - line_start  # (3,)
    
    # 计算投影
    line_len_sq = np.sum(line_vec**2, axis=1)  # (N,)
    
    # 处理极短线段
    valid = line_len_sq > 1e-12
    
    # 初始化结果数组
    result = np.zeros_like(line_len_sq)
    
    # 处理无效线段(接近点)
    if not np.all(valid):
        invalid_mask = ~valid
        result[invalid_mask] = np.sum((point - line_start)**2)
    
    if np.any(valid):
        # 计算点在线段上的投影比例
        proj = np.sum(point_vec * line_vec[valid], axis=1) / line_len_sq[valid]  # (N_valid,)
        proj = np.clip(proj, 0.0, 1.0)
        
        # 计算线段上的最近点
        closest = line_start + proj.reshape(-1, 1) * line_vec[valid]  # (N_valid,3)
        
        # 计算最近点到目标点的距离平方
        result[valid] = np.sum((closest - point)**2, axis=1)  # (N_valid,)
    
    return result

# ===== 优化版遮蔽时长计算 =====
def calculate_cover_time_fast(t_rel, dly, vF, theta):
    """优化版遮蔽时长计算，使用自适应采样"""
    t_det = t_rel + dly
    t_start = t_det
    t_end = t_det + t_window
    
    # 无人机方向向量
    uF = np.array([np.cos(theta), np.sin(theta), 0])
    
    # 预计算起爆点位置
    drop_pos = F0 + vF * t_rel * uF
    drop_vel = vF * uF
    C_det = drop_pos + drop_vel * dly + np.array([0, 0, -0.5*g*dly**2])
    
    # 粗采样
    dt_coarse = 0.1  # 粗采样步长
    ts_coarse = np.arange(t_start, t_end, dt_coarse)
    
    # 向量化视线判定
    def is_covered_batch(t_array):
        covered = np.zeros(len(t_array), dtype=bool)
        
        for i, t in enumerate(t_array):
            # 导弹位置
            missile_pos = M0 + vM * t * uM
            
            # 云团中心位置
            dt = t - t_det
            if dt < 0:
                continue
            cloud_center = C_det + np.array([0, 0, -v_sink*dt])
            
            # 计算云团中心到所有视线的距离
            dist_sq = point_to_line_segment_distance_sq_batch(cloud_center, missile_pos, cylinder_points)
            
            # 如果所有视线都在云团内，则被遮蔽
            if np.all(dist_sq <= r_cloud**2):
                covered[i] = True
                
        return covered
    
    # 粗采样评估
    covered_coarse = is_covered_batch(ts_coarse)
    
    # 找到遮蔽区间
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
                
            covered_fine = is_covered_batch(ts_fine)
            
            # 计算这个区间的遮蔽时长
            interval_time = np.sum(covered_fine) * DT
            total_time += interval_time
    
    return total_time

def _eval_particle(particle):
    """粒子适应度评估函数(可并行)"""
    theta, vF, t_rel, dly = particle
    return calculate_cover_time_fast(t_rel, dly, vF, theta)

# ===== 加速版PSO优化器 =====
class TimeoutError(Exception):
    pass

class Timeout:
    def __init__(self, seconds):
        self.seconds = seconds
        self.timer = None
        self.timeout_occurred = False
    
    def timeout_function(self):
        self.timeout_occurred = True
        thread_id = threading.current_thread().ident
        # 抛出异常会导致问题，我们只标记状态
        print(f"  [警告] 操作已超过{self.seconds}秒，标记为超时")
    
    def __enter__(self):
        self.timer = threading.Timer(self.seconds, self.timeout_function)
        self.timer.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer:
            self.timer.cancel()

class AcceleratedPSO:
    def __init__(self, n_particles=50, max_iter=200, w_init=0.9, w_end=0.4, 
                 c1_init=2.5, c1_end=0.5, c2_init=0.5, c2_end=2.5,
                 n_jobs=-1, local_search_freq=10):
        """
        加速版PSO算法
        参数:
            n_particles: 粒子数量
            max_iter: 最大迭代次数
            w_init/w_end: 惯性权重的初始/结束值
            c1_init/c1_end: 认知系数的初始/结束值
            c2_init/c2_end: 社会系数的初始/结束值
            n_jobs: 并行进程数(-1表示使用所有可用CPU)
            local_search_freq: 局部搜索频率
        """
        self.n_particles = n_particles
        self.max_iter = max_iter
        
        # 动态参数设置
        self.w_init = w_init
        self.w_end = w_end
        self.c1_init = c1_init
        self.c1_end = c1_end
        self.c2_init = c2_init
        self.c2_end = c2_end
        
        # 并行设置
        self.n_jobs = n_jobs if n_jobs > 0 else max(1, mp.cpu_count()-1)
        self.local_search_freq = local_search_freq
        
        # 搜索空间边界
        self.bounds = np.array([
            [-np.pi,  np.pi],          # theta (航向角)
            [min_speed, max_speed],    # vF (速度)
            [0.0,    20.0],            # t_rel (投放时间)
            [0.0,    20.0]             # dly (起爆延迟)
        ], dtype=float)
        
        # 智能初始化
        self.particles = self._smart_initialize(n_particles)
        
        # 初始化速度
        span = self.bounds[:,1] - self.bounds[:,0]
        self.velocities = np.random.randn(n_particles, 4) * 0.1 * span
        
        # 初始化个体最优和全局最优
        self.pbest_pos = self.particles.copy()
        self.pbest_val = np.array([-np.inf] * n_particles)
        self.gbest_pos = None
        self.gbest_val = -np.inf
        self.elite_pos = None
        self.elite_val = -np.inf
        
        # 记录历史最优值
        self.history = []
        self.iter_nums = []
    
    def _smart_initialize(self, n_particles):
        """智能初始化粒子位置 (采用A_3pso的初始化策略)"""
        particles = np.zeros((n_particles, 4))
        
        # 20% 完全随机初始化
        num_random = int(n_particles * 0.2)
        for i in range(num_random):
            for j in range(4):
                particles[i, j] = self.bounds[j, 0] + np.random.random() * (self.bounds[j, 1] - self.bounds[j, 0])
        
        # 40% 使用均匀分布的航向角和参数组合
        num_preset = int(n_particles * 0.4)
        if num_preset > 0:
            # 均匀角度分布
            angles = np.linspace(-np.pi, np.pi, num_preset, endpoint=False)
            for i in range(num_random, num_random + num_preset):
                idx = i - num_random
                particles[i, 0] = angles[idx] + np.random.normal(0, 0.05)
                # 投放时间从较早开始
                particles[i, 2] = np.random.uniform(0, 5)
                # 速度在中等偏快范围
                particles[i, 1] = np.random.uniform(80, 120)
                # 延迟时间中等范围
                particles[i, 3] = np.random.uniform(1, 4)
        
        # 其余粒子随机初始化但偏向小角度探索
        for i in range(num_random + num_preset, n_particles):
            particles[i, 0] = np.random.normal(0, 0.5)  # 角度集中在0附近
            for j in range(1, 4):
                particles[i, j] = self.bounds[j, 0] + np.random.random() * (self.bounds[j, 1] - self.bounds[j, 0])
        
        return particles
    
    def get_dynamic_params(self, iter):
        """获取动态参数(与A_3pso一致)"""
        progress = iter / self.max_iter
        w = self.w_init - (self.w_init - self.w_end) * progress
        # 简化为使用恒定值
        c1 = self.c1_init
        c2 = self.c2_init
        return w, c1, c2
    
    # 替换原来的局部搜索函数
    def local_search(self):
        """在全局最优附近进行局部搜索（Windows兼容版，采用A_3pso的搜索策略）"""
        if self.gbest_val <= 0:
            return
        
        print("  开始局部搜索...")
        
        # 减少局部搜索点数
        n_local = 10
        best_val = self.gbest_val
        best_pos = self.gbest_pos.copy()
        
        # 局部搜索范围随迭代减小(与A_3pso一致)
        prog = len(self.history) / self.max_iter
        scale = 0.2 * (1 - prog) + 0.02
        
        # 准备搜索点
        search_points = []
        
        # 对每个维度单独扰动(与A_3pso一致)
        span = self.bounds[:,1] - self.bounds[:,0]
        for dim in range(4):
            # 每个维度正负方向各尝试一点
            delta = span[dim] * scale * 0.1
            
            # 正向扰动
            point_pos = best_pos.copy()
            point_pos[dim] += delta
            point_pos[dim] = min(self.bounds[dim, 1], point_pos[dim])
            search_points.append(point_pos)
            
            # 负向扰动
            point_neg = best_pos.copy()
            point_neg[dim] -= delta
            point_neg[dim] = max(self.bounds[dim, 0], point_neg[dim])
            search_points.append(point_neg)
        
        # 再增加几个随机点
        for _ in range(n_local - 8):
            # 随机扰动幅度更小(与A_3pso一致)
            point = best_pos + np.random.normal(0, 1, 4) * scale * 0.05 * span
            # 边界处理
            for j in range(4):
                point[j] = max(self.bounds[j, 0], min(self.bounds[j, 1], point[j]))
            search_points.append(point)
        
        search_points = np.array(search_points)
        
        try:
            with Timeout(30):
                # 使用临时进程池避免资源泄漏
                with mp.Pool(processes=min(self.n_jobs, len(search_points))) as pool:
                    local_fitness = np.array(pool.map(_eval_particle, search_points))
                
                # 更新全局最优
                best_idx = np.argmax(local_fitness)
                if local_fitness[best_idx] > best_val:
                    best_val = local_fitness[best_idx]
                    best_pos = search_points[best_idx].copy()
                    print(f"  [局部搜索] 发现更优解: {best_val:.6f}s")
                    
                    # 更新全局最优
                    if best_val > self.gbest_val:
                        self.gbest_val = best_val
                        self.gbest_pos = best_pos.copy()
        except Exception as e:
            print(f"  [错误] 局部搜索异常: {str(e)}")
    
    def optimize(self):
        """执行优化"""
        print(f"开始并行PSO优化，使用{self.n_jobs}个CPU核心...")
        print(f"粒子数: {self.n_particles}, 最大迭代: {self.max_iter}")
        
        # 并行评估初始种群
        with mp.Pool(processes=self.n_jobs) as pool:
            self.pbest_val = np.array(pool.map(_eval_particle, self.particles))
        
        self.pbest_pos = self.particles.copy()
        best_idx = np.argmax(self.pbest_val)
        self.gbest_pos = self.pbest_pos[best_idx].copy()
        self.gbest_val = self.pbest_val[best_idx]
        
        # 记录初始最优
        self.history.append(self.gbest_val)
        self.iter_nums.append(0)
        print(f"[PSO 0/{self.max_iter}] gbest={self.gbest_val:.6f}s")
        
        # 主迭代循环
        for iter in range(1, self.max_iter + 1):
            # 获取动态参数
            w, c1, c2 = self.get_dynamic_params(iter)
            
            # 更新速度和位置
            r1, r2 = np.random.rand(self.n_particles, 4), np.random.rand(self.n_particles, 4)
            cognitive = c1 * r1 * (self.pbest_pos - self.particles)
            social = c2 * r2 * (self.gbest_pos - self.particles)
            self.velocities = w * self.velocities + cognitive + social
            
            # 速度限制
            span = self.bounds[:,1] - self.bounds[:,0]
            self.velocities = np.clip(self.velocities, -0.2*span, 0.2*span)
            
            # 更新位置
            self.particles += self.velocities
            
            # 边界处理
            for j in range(4):
                self.particles[:, j] = np.clip(self.particles[:, j], 
                                             self.bounds[j, 0], 
                                             self.bounds[j, 1])
            
            # 并行评估
            with mp.Pool(processes=self.n_jobs) as pool:
                fitness = np.array(pool.map(_eval_particle, self.particles))
            
            # 更新个体最优
            improved = fitness > self.pbest_val
            self.pbest_pos[improved] = self.particles[improved]
            self.pbest_val[improved] = fitness[improved]
            
            # 更新全局最优
            best_idx = np.argmax(self.pbest_val)
            iter_best = np.max(fitness)
            if self.pbest_val[best_idx] > self.gbest_val:
                self.gbest_val = self.pbest_val[best_idx]
                self.gbest_pos = self.pbest_pos[best_idx].copy()
            
            # 精英保留
            if self.gbest_val > self.elite_val:
                self.elite_val = self.gbest_val
                self.elite_pos = self.gbest_pos.copy()
            
            # 记录历史最优值
            self.history.append(self.gbest_val)
            self.iter_nums.append(iter)
            
            print(f"[PSO {iter}/{self.max_iter}] iter_best={iter_best:.6f}s, gbest={self.gbest_val:.6f}s")
            
            # 每10次迭代进行局部搜索
            if iter % self.local_search_freq == 0:
                try:
                    self.local_search()
                except Exception as e:
                    print(f"  [警告] 局部搜索异常: {str(e)}")
        
        # 替换最终局部搜索调用
        print("执行最终精细局部搜索...")
        try:
            self.local_search()
        except Exception as e:
            print(f"[警告] 最终局部搜索失败: {str(e)}, 使用当前最优解")
        
        # 使用精英解
        if self.elite_val > self.gbest_val:
            self.gbest_val = self.elite_val
            self.gbest_pos = self.elite_pos.copy()
        
        # 最后绘制完整收敛曲线
        self.plot_convergence()
        
        return self.gbest_pos, self.gbest_val
    
    def plot_convergence(self):
        """绘制完整收敛曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"A2-PSO收敛曲线", fontsize=14)
        ax.set_xlabel("迭代次数", fontsize=12)
        ax.set_ylabel("遮蔽时长 (s)", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.7)
        
        # 绘制收敛曲线
        ax.plot(self.iter_nums, self.history, "-o", ms=4, color="#FF7F0E", 
                linewidth=1.8, label="全局最优值")
        
        # 在最终点标记
        ax.scatter([self.iter_nums[-1]], [self.history[-1]], s=100, 
                   marker="*", color="#C44E52", label=f"最终值: {self.history[-1]:.4f}s")
        
        # 添加改进点标注
        improvements = np.diff(self.history) > 0
        for i in range(1, len(self.iter_nums)):
            if improvements[i-1] and i % max(1, len(self.iter_nums)//10) == 0:
                ax.annotate(f"{self.history[i]:.2f}", 
                           (self.iter_nums[i], self.history[i]),
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=9,
                           arrowprops=dict(arrowstyle='->', color='gray'))
        
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("pso_q2_convergence_plot.png", dpi=300, bbox_inches='tight')
        plt.show()

# ===== 主函数 =====
def main():
    t0 = time.time()
    
    # 使用A_3pso中的优化参数
    pso = AcceleratedPSO(
        n_particles=60,              # 从A_3pso.py中采用(原来也是60)
        max_iter=250,                # 从200增加到250(与A_3pso.py一致)
        w_init=0.9, w_end=0.4,       # 保持不变(与A_3pso.py一致)
        c1_init=1.8, c1_end=1.8,     # 简化为恒定值1.8(与A_3pso.py一致)
        c2_init=1.8, c2_end=1.8,     # 简化为恒定值1.8(与A_3pso.py一致)
        n_jobs=-1,                   # 使用所有可用核心
        local_search_freq=10         # 每10次迭代进行局部搜索(与A_3pso.py一致)
    )
    
    best_params, best_score = pso.optimize()
    elapsed = time.time() - t0
    
    # 解析最佳参数
    theta_opt, vF_opt, t_rel_opt, dly_opt = best_params
    uF = np.array([np.cos(theta_opt), np.sin(theta_opt), 0])
    
    # 投放点
    drop_point = F0 + vF_opt * t_rel_opt * uF
    
    # 起爆点
    drop_vel = vF_opt * uF
    bomb_pos = drop_point + drop_vel * dly_opt + np.array([0, 0, -0.5 * g * dly_opt**2])
    
    print(f"\n===== 最优结果 =====")
    print(f"航向角θ: {np.degrees(theta_opt):.4f}°")
    print(f"速度v_F: {vF_opt:.4f} m/s")
    print(f"投放时刻: {t_rel_opt:.4f} s")
    print(f"起爆延迟: {dly_opt:.4f} s")
    print(f"起爆时刻: {(t_rel_opt + dly_opt):.4f} s")
    print(f"总遮蔽时长: {best_score:.6f} s")
    print(f"优化用时: {elapsed:.2f} s")
    
    # 可视化最优解
    visualize_solution(theta_opt, vF_opt, t_rel_opt, dly_opt, drop_point, bomb_pos, best_score)
    
    # 保存结果
    save_results(best_params, best_score, drop_point, bomb_pos)

# ===== 可视化函数 =====
def visualize_solution(theta, vF, t_rel, dly, drop_point, bomb_pos, best_score):
    """可视化最优解下的遮蔽效果（简化版）"""
    # 计算关键参数
    t_det = t_rel + dly
    t_end = t_det + t_window
    uF = np.array([np.cos(theta), np.sin(theta), 0])
    
    # 创建遮蔽状态图
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # 准备时间轴数据
    ts = np.linspace(t_det - 1, t_end + 1, 300)  # 精细采样
    
    # 计算每个时刻的遮蔽状态
    covered_status = []
    for t in ts:
        # 导弹位置
        missile_pos = M0 + vM * t * uM
        
        # 云团中心位置（仅在t≥t_det时存在）
        if t < t_det:
            covered_status.append(0.0)
            continue
            
        dt = t - t_det
        if dt > t_window:
            covered_status.append(0.0)
            continue
            
        cloud_center = bomb_pos + np.array([0, 0, -v_sink * dt])
        
        # 计算是否遮蔽
        dist_sq = point_to_line_segment_distance_sq_batch(cloud_center, missile_pos, cylinder_points)
        covered_status.append(1.0 if np.all(dist_sq <= r_cloud**2) else 0.0)
    
    covered_status = np.array(covered_status)
    
    # 寻找遮蔽区间
    intervals = []
    if covered_status.any():
        # 找到状态变化点
        changes = np.diff(covered_status)
        start_idxs = np.where(changes > 0.5)[0]
        end_idxs = np.where(changes < -0.5)[0]
        
        # 处理边界情况
        if covered_status[0] > 0.5:
            start_idxs = np.insert(start_idxs, 0, -1)
        if covered_status[-1] > 0.5:
            end_idxs = np.append(end_idxs, len(covered_status)-1)
        
        # 配对并获取实际时间
        for start_i, end_i in zip(start_idxs, end_idxs):
            start_t = ts[start_i+1] if start_i >= 0 else ts[0]
            end_t = ts[end_i] if end_i < len(ts) else ts[-1]
            intervals.append((start_t, end_t))
    
    # 绘制遮蔽状态函数
    ax.plot(ts, covered_status, color="#1f77b4", linewidth=2, label="遮蔽状态")
    
    # 填充遮蔽区间
    ax.fill_between(ts, 0, 1, where=covered_status > 0.5,
                   alpha=0.3, color="#add8e6", label="遮蔽区间")
    
    # 标记关键时间点
    ax.axvline(t_rel, linestyle="--", color="green", alpha=0.7, label="投放时刻")
    ax.axvline(t_det, linestyle="--", color="red", alpha=0.7, label="起爆时刻")
    ax.axvline(t_end, linestyle="--", color="purple", alpha=0.7, label="有效期结束")
    
    # 设置轴标签和标题
    ax.set_xlabel("时间 t (秒)", fontsize=12)
    ax.set_ylabel("遮蔽状态 (1=遮蔽, 0=未遮蔽)", fontsize=12)
    ax.set_title("问题二最优解的遮蔽状态函数", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":")
    
    # 计算总遮蔽时长
    total_covered = sum(end-start for start, end in intervals)
    
    # 在图中添加关键参数信息
    info_text = (
        f"航向角: {np.degrees(theta):.4f}°\n"
        f"飞行速度: {vF:.4f} m/s\n"
        f"投放时刻: {t_rel:.4f} s\n"
        f"起爆延迟: {dly:.4f} s\n"
        f"总遮蔽时长: {best_score:.4f} s"  # 使用优化算法给出的确切值
    )
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig("problem2_coverage.png", dpi=300, bbox_inches='tight')
    plt.show()

def save_results(params, score, drop_point, bomb_pos):
    """保存结果到Excel文件"""
    theta, vF, t_rel, dly = params
    t_det = t_rel + dly
    
    data = {
        '参数': [
            '航向角(弧度)', '航向角(度)', 
            '速度(m/s)', 
            '投放时刻(s)', '起爆延迟(s)', '起爆时刻(s)', 
            '遮蔽时长(s)',
            '投放点x(m)', '投放点y(m)', '投放点z(m)',
            '起爆点x(m)', '起爆点y(m)', '起爆点z(m)'
        ],
        '值': [
            theta, np.degrees(theta),
            vF,
            t_rel, dly, t_det,
            score,
            drop_point[0], drop_point[1], drop_point[2],
            bomb_pos[0], bomb_pos[1], bomb_pos[2]
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_excel('result_problem2.xlsx', index=False)
    print("问题二结果已保存到 result_problem2.xlsx")

if __name__ == "__main__":
    main()