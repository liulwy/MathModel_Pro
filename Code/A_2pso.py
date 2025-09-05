import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===== 常量定义 =====
# 假目标和真目标位置
fake_target = np.array([0.0, 0.0, 0.0])  # 假目标位置
T_center = np.array([0.0, 200.0, 5.0])    # 真目标圆柱体中心
T_radius = 7.0    # 圆柱体半径(m)
T_height = 10.0   # 圆柱体高度(m)

# 导弹参数
M0 = np.array([20000.0, 0.0, 2000.0])  # M1初始位置
vM = 300.0  # 导弹速度(m/s)
# 导弹方向向量 (指向假目标)
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
    """生成圆柱体表面采样点"""
    points = []
    z_min = T_center[2] - T_height/2
    z_max = T_center[2] + T_height/2
    
    # 侧面点
    theta = np.linspace(0, 2*np.pi, int(np.sqrt(num_points)), endpoint=False)
    z = np.linspace(z_min, z_max, int(np.sqrt(num_points)))
    for t in theta:
        for zi in z:
            points.append(np.array([
                T_center[0] + T_radius * np.cos(t),
                T_center[1] + T_radius * np.sin(t),
                zi
            ]))
    
    # 顶面点
    r = np.linspace(0, T_radius, int(np.sqrt(num_points/2)))
    theta = np.linspace(0, 2*np.pi, int(np.sqrt(num_points/2)), endpoint=False)
    for ri in r:
        for t in theta:
            points.append(np.array([
                T_center[0] + ri * np.cos(t),
                T_center[1] + ri * np.sin(t),
                z_max
            ]))
    
    return points[:num_points]

cylinder_points = generate_cylinder_points(100)

# ===== 几何计算函数 =====
def is_line_of_sight_blocked(cloud_center, missile_pos, target_point):
    """检查烟幕是否阻挡了导弹到目标点的视线"""
    # 导弹到目标的向量
    missile_to_target = target_point - missile_pos
    # 导弹到云团中心的向量
    missile_to_cloud = cloud_center - missile_pos
    
    # 计算投影比例
    t = np.dot(missile_to_cloud, missile_to_target) / np.dot(missile_to_target, missile_to_target)
    t = max(0.0, min(1.0, t))  # 钳制到[0,1]区间
    
    # 计算视线上的最近点
    closest_point = missile_pos + t * missile_to_target
    
    # 检查云团中心到最近点的距离
    distance = np.linalg.norm(cloud_center - closest_point)
    return distance <= r_cloud

# ===== 遮蔽时长计算 =====
def calculate_cover_time(t_rel, dly, vF, theta):
    """计算遮蔽时长"""
    t_det = t_rel + dly
    t_start = t_det
    t_end = t_det + t_window
    
    # 无人机方向向量
    uF = np.array([np.cos(theta), np.sin(theta), 0])
    
    # 无人机位置函数 (等高度平飞)
    def F(t):
        return F0 + vF * t * uF
    
    # 干扰弹位置函数 (投放后只受重力影响)
    def B(t):
        if t < t_rel:
            return F(t)
        dt = t - t_rel
        # 投放点位置和速度
        drop_pos = F(t_rel)
        drop_vel = vF * uF  # 投放瞬间只有水平速度
        # 干扰弹位置 (水平匀速，垂直自由落体)
        return drop_pos + drop_vel * dt + np.array([0, 0, -0.5*g*dt**2])
    
    # 云团中心位置函数
    C_det = B(t_det)  # 起爆点位置
    def C(t):
        if t < t_det:
            return np.array([np.nan, np.nan, np.nan])
        dt = t - t_det
        return C_det + np.array([0, 0, -v_sink*dt])
    
    # 导弹位置函数
    def M(t):
        return M0 + vM * t * uM
    
    # 检查特定时间点是否被遮蔽
    def is_covered(t):
        cloud_center = C(t)
        if any(np.isnan(cloud_center)):
            return False
        
        missile_pos = M(t)
        
        # 检查圆柱体表面所有点是否至少有一个视线被阻挡
        for point in cylinder_points:
            if not is_line_of_sight_blocked(cloud_center, missile_pos, point):
                return False
        return True
    
    # 时间采样
    ts = np.arange(t_start, t_end, DT)
    covered = np.array([is_covered(t) for t in ts])
    return np.sum(covered) * DT

# ===== 改进的粒子群优化算法 =====
class EnhancedPSO:
    def __init__(self, n_particles=50, max_iter=200, w_init=0.9, w_end=0.4, 
                 c1_init=2.5, c1_end=0.5, c2_init=0.5, c2_end=2.5):
        """
        改进的PSO算法
        参数:
            n_particles: 粒子数量
            max_iter: 最大迭代次数
            w_init/w_end: 惯性权重的初始/结束值
            c1_init/c1_end: 认知系数的初始/结束值
            c2_init/c2_end: 社会系数的初始/结束值
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
        
        # 搜索空间边界 (与 a2.py 保持一致)
        self.bounds = np.array([
            [-np.pi,  np.pi],          # theta (航向角)
            [min_speed, max_speed],    # vF (速度)
            [0.0,    20.0],            # t_rel (投放时间)
            [0.0,    20.0]             # dly (起爆延迟)
        ], dtype=float)
        
        # 初始化粒子
        self.particles = np.random.rand(n_particles, 4)
        for i in range(4):
            self.particles[:, i] = self.bounds[i, 0] + self.particles[:, i] * (self.bounds[i, 1] - self.bounds[i, 0])
        
        # 初始化速度
        self.velocities = np.random.randn(n_particles, 4) * 0.1
        
        # 初始化个体最优
        self.pbest_pos = self.particles.copy()
        self.pbest_val = np.array([-np.inf] * n_particles)
        
        # 初始化全局最优
        self.gbest_pos = None
        self.gbest_val = -np.inf
        
        # 记录历史最优值
        self.history = []
        
        # 精英保留
        self.elite_pos = None
        self.elite_val = -np.inf
    
    def evaluate(self, particle):
        """评估粒子的适应度"""
        theta, vF, t_rel, dly = particle
        return calculate_cover_time(t_rel, dly, vF, theta)
    
    def get_dynamic_params(self, iter):
        """获取动态参数"""
        progress = iter / self.max_iter
        w = self.w_init - (self.w_init - self.w_end) * progress
        c1 = self.c1_init - (self.c1_init - self.c1_end) * progress
        c2 = self.c2_init + (self.c2_end - self.c2_init) * progress
        return w, c1, c2
    
    def optimize(self):
        """执行优化"""
        print(f"开始改进PSO优化，粒子数: {self.n_particles}, 最大迭代: {self.max_iter}")
        
        for iter in tqdm(range(self.max_iter), desc="PSO优化进度"):
            # 获取动态参数
            w, c1, c2 = self.get_dynamic_params(iter)
            
            # 评估所有粒子
            for i in range(self.n_particles):
                fitness = self.evaluate(self.particles[i])
                
                # 更新个体最优
                if fitness > self.pbest_val[i]:
                    self.pbest_val[i] = fitness
                    self.pbest_pos[i] = self.particles[i].copy()
                
                # 更新全局最优
                if fitness > self.gbest_val:
                    self.gbest_val = fitness
                    self.gbest_pos = self.particles[i].copy()
            
            # 精英保留
            if self.gbest_val > self.elite_val:
                self.elite_val = self.gbest_val
                self.elite_pos = self.gbest_pos.copy()
            
            # 记录历史最优值
            self.history.append(self.gbest_val)
            
            # 更新速度和位置
            for i in range(self.n_particles):
                # 计算新速度
                r1, r2 = np.random.rand(2)
                cognitive = c1 * r1 * (self.pbest_pos[i] - self.particles[i])
                social = c2 * r2 * (self.gbest_pos - self.particles[i])
                self.velocities[i] = w * self.velocities[i] + cognitive + social
                
                # 更新位置
                self.particles[i] += self.velocities[i]
                
                # 边界检查
                for j in range(4):
                    if self.particles[i, j] < self.bounds[j, 0]:
                        self.particles[i, j] = self.bounds[j, 0]
                        self.velocities[i, j] *= -0.5
                    elif self.particles[i, j] > self.bounds[j, 1]:
                        self.particles[i, j] = self.bounds[j, 1]
                        self.velocities[i, j] *= -0.5
            
            # 每20次迭代进行一次局部搜索
            if (iter + 1) % 20 == 0:
                self.local_search()
        
        # 最终使用精英解
        if self.elite_val > self.gbest_val:
            self.gbest_val = self.elite_val
            self.gbest_pos = self.elite_pos.copy()
        
        print("\n优化完成!")
        print(f"最优航向角: {np.degrees(self.gbest_pos[0]):.4f}°")
        print(f"最优速度: {self.gbest_pos[1]:.4f} m/s")
        print(f"最优投放时刻: {self.gbest_pos[2]:.4f} s")
        print(f"最优起爆延迟: {self.gbest_pos[3]:.4f} s")
        print(f"最大遮蔽时长: {self.gbest_val:.6f} s")
        
        return self.gbest_pos, self.gbest_val
    
    def local_search(self):
        """在全局最优附近进行局部搜索"""
        if self.gbest_val <= 0:
            return
        
        # 局部搜索范围
        local_range = np.array([0.05, 2.0, 0.1, 0.1])
        
        # 生成局部搜索点
        n_local = 20
        local_points = np.random.randn(n_local, 4)
        for i in range(4):
            local_points[:, i] = self.gbest_pos[i] + local_points[:, i] * local_range[i]
            local_points[:, i] = np.clip(local_points[:, i], self.bounds[i, 0], self.bounds[i, 1])
        
        # 评估局部点
        for point in local_points:
            fitness = self.evaluate(point)
            if fitness > self.gbest_val:
                self.gbest_val = fitness
                self.gbest_pos = point.copy()
    
    def plot_convergence(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history, 'b-', linewidth=2)
        plt.xlabel('迭代次数', fontsize=12)
        plt.ylabel('遮蔽时长 (s)', fontsize=12)
        plt.title('改进PSO优化收敛曲线', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ===== 多次独立运行 =====
def multi_run_PSO(n_runs=3):
    """多次独立运行PSO以避免局部最优"""
    best_solutions = []
    
    for run in range(n_runs):
        print(f"\n=== 第 {run+1}/{n_runs} 次独立运行 ===")
        pso = EnhancedPSO(n_particles=50, max_iter=200)
        best_params, best_score = pso.optimize()
        best_solutions.append((best_params, best_score))
        pso.plot_convergence()
    
    # 选择多次运行中的最优解
    best_idx = np.argmax([s[1] for s in best_solutions])
    return best_solutions[best_idx]

# ===== 主函数 =====
def main():
    # 运行改进的PSO算法
    best_params, best_score = multi_run_PSO()
    
    # 计算关键点位置
    theta_opt, vF_opt, t_rel_opt, dly_opt = best_params
    uF = np.array([np.cos(theta_opt), np.sin(theta_opt), 0])
    
    # 投放点 (无人机在投放时刻的位置)
    drop_point = F0 + vF_opt * t_rel_opt * uF
    
    # 起爆点 (干扰弹在起爆时刻的位置)
    # 投放后只受重力影响，水平速度保持不变
    drop_vel = vF_opt * uF
    bomb_pos = drop_point + drop_vel * dly_opt + np.array([0, 0, -0.5 * g * dly_opt**2])
    
    # 可视化最优解
    visualize_solution(theta_opt, vF_opt, t_rel_opt, dly_opt, drop_point, bomb_pos)
    
    # 保存结果
    save_results(best_params, best_score, drop_point, bomb_pos)

# ===== 可视化函数 =====
def visualize_solution(theta, vF, t_rel, dly, drop_point, bomb_pos):
    """可视化最优解下的轨迹和遮蔽效果"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 无人机方向向量
    uF = np.array([np.cos(theta), np.sin(theta), 0])
    
    # 无人机位置函数 (等高度平飞)
    def F(t):
        return F0 + vF * t * uF
    
    # 干扰弹位置函数 (投放后只受重力影响)
    def B(t):
        if t < t_rel:
            return F(t)
        dt = t - t_rel
        drop_vel = vF * uF
        return drop_point + drop_vel * dt + np.array([0, 0, -0.5*g*dt**2])
    
    # 云团中心位置函数
    t_det = t_rel + dly
    C_det = B(t_det)
    def C(t):
        if t < t_det:
            return np.array([np.nan, np.nan, np.nan])
        dt = t - t_det
        return C_det + np.array([0, 0, -v_sink*dt])
    
    # 导弹位置函数
    def M(t):
        return M0 + vM * t * uM
    
    # 计算轨迹
    t_max = max(30, t_det + t_window)
    ts = np.linspace(0, t_max, 100)
    
    missile_traj = np.array([M(t) for t in ts])
    drone_traj = np.array([F(t) for t in ts])
    bomb_traj = np.array([B(t) for t in ts if t >= t_rel])
    cloud_traj = np.array([C(t) for t in ts if t >= t_det])
    
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
    ax.scatter(*drop_point, c='green', s=60, label='投放点')
    ax.scatter(*bomb_pos, c='magenta', s=100, marker='*', label='起爆点')
    
    # 绘制假目标和真目标
    ax.scatter(*fake_target, c='orange', s=100, marker='s', label='假目标')
    
    # 绘制真目标圆柱体
    z_min = T_center[2] - T_height/2
    z_max = T_center[2] + T_height/2
    
    # 绘制侧面
    theta_vals = np.linspace(0, 2 * np.pi, 50)
    z_vals = np.linspace(z_min, z_max, 10)
    theta_grid, z_grid = np.meshgrid(theta_vals, z_vals)
    x = T_center[0] + T_radius * np.cos(theta_grid)
    y = T_center[1] + T_radius * np.sin(theta_grid)
    ax.plot_surface(x, y, z_grid, color='gray', alpha=0.3, label='真目标')
    
    # 绘制顶面
    r_vals = np.linspace(0, T_radius, 10)
    theta_top = np.linspace(0, 2 * np.pi, 50)
    r_grid, theta_top_grid = np.meshgrid(r_vals, theta_top)
    x_top = T_center[0] + r_grid * np.cos(theta_top_grid)
    y_top = T_center[1] + r_grid * np.sin(theta_top_grid)
    z_top = np.full_like(x_top, z_max)
    ax.plot_surface(x_top, y_top, z_top, color='gray', alpha=0.3)
    
    # 添加文字标注
    ax.text(T_center[0], T_center[1], T_center[2], "真目标中心", fontsize=12, color='black', zorder=10)
    
    # 设置图形属性
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('最优投放策略几何示意图', fontsize=14)
    ax.legend(fontsize=10)
    
    # 调整视角
    ax.view_init(elev=25, azim=-45)
    
    # 设置坐标轴范围
    ax.set_xlim([-1000, 21000])
    ax.set_ylim([-1000, 2100])
    ax.set_zlim([-100, 2500])
    
    plt.tight_layout()
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