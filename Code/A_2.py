# 问题二优化框架（PSO 粒子群优化）
import numpy as np
import time
import matplotlib.pyplot as plt  # 在线可视化

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===== 圆柱体目标参数（替换原轴线采样）=====
T_center = np.array([0.0, 200.0, 5.0], dtype=float)  # 圆柱体中心
T_radius = 7.0                                        # 圆柱体半径(m)
T_height = 10.0                                       # 圆柱体高度(m)

M0 = np.array([20000.0, 0.0, 2000.0])  # 导弹初始位置
F0 = np.array([17800.0, 0.0, 1800.0])  # 无人机初始位置
vM = 300.0  # 导弹速度
g = 9.8  # 重力加速度
v_sink = 3.0  # 云团下沉速度
r_cloud = 10.0  # 云团有效半径
t_window = 20.0  # 云团有效时间

# 离散时间步长
DT = 0.02  # 秒（保留较小步长以保证时域积分精度）

# ===== 圆柱体表面采样（侧面+顶面，不含底面）=====
def generate_cylinder_points(num_points=100):
    """在圆柱体表面生成近似等面积分布的点（不包括底面）"""
    points = []

    z_min = T_center[2] - T_height / 2
    z_max = T_center[2] + T_height / 2

    area_side = 2 * np.pi * T_radius * T_height
    area_top = np.pi * T_radius**2
    total_area = area_side + area_top

    num_side = int(num_points * area_side / total_area)
    num_top = max(0, num_points - num_side)

    # 侧面 z-θ 规则网格（近似等面积）
    num_z = max(3, int(np.sqrt(max(num_side, 1) / (2 * np.pi))))
    num_theta = max(6, int(max(num_side, 1) / max(num_z, 1)))
    z_values = np.linspace(z_min, z_max, num_z)
    theta_values = np.linspace(0, 2*np.pi, num_theta, endpoint=False)
    for z in z_values:
        for theta in theta_values:
            x = T_center[0] + T_radius * np.cos(theta)
            y = T_center[1] + T_radius * np.sin(theta)
            points.append(np.array([x, y, z], dtype=float))

    # 顶面极坐标网格（简化近似）
    num_r = max(2, int(np.sqrt(max(num_top, 1))))
    num_theta_top = max(6, int(max(num_top, 1) / max(num_r, 1)))
    r_values = np.linspace(0, T_radius, num_r)
    theta_values_top = np.linspace(0, 2*np.pi, num_theta_top, endpoint=False)
    for r in r_values:
        for theta in theta_values_top:
            x = T_center[0] + r * np.cos(theta)
            y = T_center[1] + r * np.sin(theta)
            points.append(np.array([x, y, z_max], dtype=float))

    return points[:num_points]

# 生成圆柱体表面点（不包括底面）
cylinder_points = generate_cylinder_points(100)

def covered_time(t_rel, dly, vF, theta):
    """计算给定参数下(圆柱面采样，全视线判定)的遮蔽时长"""
    t_det = t_rel + dly
    t_start = t_det
    t_end = t_det + t_window

    # 无人机方向向量
    uF = np.array([np.cos(theta), np.sin(theta), 0.0])

    # 无人机位置函数
    def F(t):
        return F0 + vF * t * uF

    # 干扰弹位置函数
    def B(t):
        if t < t_rel:
            return F(t)
        dt = t - t_rel
        rel_pos = F(t_rel)
        return rel_pos + vF * uF * dt + np.array([0.0, 0.0, -0.5 * g * dt**2])

    # 云团中心位置函数
    C_det = B(t_det)
    def C(t):
        if t < t_det:
            return np.array([np.nan, np.nan, np.nan])
        dt = t - t_det
        return C_det + np.array([0.0, 0.0, -v_sink * dt])

    # 导弹位置函数
    uM = -M0 / np.linalg.norm(M0)
    def M(t):
        return M0 + vM * t * uM

    # 点到线段最短距离平方
    def point_to_line_segment_distance_sq(point, line_start, line_end):
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len_sq = float(np.dot(line_vec, line_vec))
        if line_len_sq < 1e-12:
            return float(np.dot(point_vec, point_vec))
        s = float(np.dot(point_vec, line_vec) / line_len_sq)
        s = max(0.0, min(1.0, s))
        proj = line_start + s * line_vec
        diff = point - proj
        return float(np.dot(diff, diff))

    # 单时刻全覆盖判定（任一射线超阈即失败）
    thr2 = r_cloud**2 + 1e-12
    def is_covered(t):
        Ct = C(t)
        if np.isnan(Ct).any():
            return False
        Mt = M(t)
        for Ti in cylinder_points:
            d2 = point_to_line_segment_distance_sq(Ct, Mt, Ti)
            if d2 > thr2:
                return False
        return True

    # 时间采样并积分
    ts = np.arange(t_start, t_end, DT)
    covered_flags = np.fromiter((is_covered(t) for t in ts), dtype=bool, count=ts.size)
    return float(np.sum(covered_flags) * DT)

# ==================== 改进 PSO 优化器（动态权重/系数 + 精英 + 局部搜索） ====================
def angle_wrap(theta):
    return (theta + np.pi) % (2*np.pi) - np.pi

class EnhancedPSO:
    def __init__(
        self,
        swarm_size=50,
        max_iters=300,
        w_start=0.9, w_end=0.4,
        c1_start=2.5, c1_end=0.5,
        c2_start=0.5, c2_end=2.5,
        seed=None
    ):
        self.rng = np.random.default_rng(seed)
        self.swarm_size = swarm_size
        self.max_iters = max_iters
        self.w_start, self.w_end = w_start, w_end
        self.c1_start, self.c1_end = c1_start, c1_end
        self.c2_start, self.c2_end = c2_start, c2_end

        # 决策变量: [theta, vF, t_rel, dly]
        self.lb = np.array([-np.pi, 70.0, 0.0, 0.0], dtype=float)
        self.ub = np.array([ np.pi,140.0,10.0, 6.0], dtype=float)
        span = self.ub - self.lb

        # 初始化
        self.X = self.lb + self.rng.random((swarm_size, 4)) * span
        self.X[:, 0] = np.array([angle_wrap(t) for t in self.X[:, 0]])
        self.vmax = 0.2 * span
        self.V = self.rng.uniform(-self.vmax, self.vmax, size=(swarm_size, 4))

        # 适应度
        self.fitness = np.zeros(swarm_size, dtype=float)
        self.pbest = self.X.copy()
        self.pbest_fit = np.full(swarm_size, -np.inf, dtype=float)
        self.gbest = None
        self.gbest_fit = -np.inf

        # 收敛历史
        self.it_hist = []
        self.best_hist = []

    def eval_fit(self, x):
        th, v, tr, dl = x
        return covered_time(tr, dl, v, th)

    def anneal_params(self, it):
        r = it / max(1, self.max_iters)
        w = self.w_start + (self.w_end - self.w_start) * r
        c1 = self.c1_start + (self.c1_end - self.c1_start) * r
        c2 = self.c2_start + (self.c2_end - self.c2_start) * r
        return w, c1, c2

    def local_search(self):
        if self.gbest is None:
            return
        # 在全局最优附近做小扰动搜索
        ranges = np.array([0.05, 2.0, 0.1, 0.1], dtype=float)
        for _ in range(20):
            cand = self.gbest + self.rng.normal(scale=ranges, size=4)
            cand = np.clip(cand, self.lb, self.ub)
            cand[0] = angle_wrap(cand[0])
            fit = self.eval_fit(cand)
            if fit > self.gbest_fit:
                self.gbest_fit = fit
                self.gbest = cand

    def optimize(self):
        # 在线收敛曲线
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("PSO 收敛：遮掩时间 vs 迭代次数（best-so-far）")
        ax.set_xlabel("迭代次数")
        ax.set_ylabel("遮掩时间 (s)")
        ax.grid(True, linestyle=":")
        line, = ax.plot([], [], "-o", ms=3, color="#FF7F0E")  # 橙色

        # 初次评估
        self.fitness = np.array([self.eval_fit(x) for x in self.X], dtype=float)
        self.pbest = self.X.copy()
        self.pbest_fit = self.fitness.copy()
        g_idx = int(np.argmax(self.pbest_fit))
        self.gbest = self.pbest[g_idx].copy()
        self.gbest_fit = float(self.pbest_fit[g_idx])

        self.it_hist = [0]
        self.best_hist = [self.gbest_fit]
        line.set_data(self.it_hist, self.best_hist)
        ax.relim(); ax.autoscale_view()
        fig.canvas.draw(); fig.canvas.flush_events()
        print(f"[PSO 0/{self.max_iters}] iter_best={np.max(self.fitness):.6f}s, gbest={self.gbest_fit:.6f}s")

        for it in range(1, self.max_iters + 1):
            w, c1, c2 = self.anneal_params(it)
            r1 = self.rng.random((self.swarm_size, 4))
            r2 = self.rng.random((self.swarm_size, 4))

            # 速度与位置更新
            self.V = w * self.V + c1 * r1 * (self.pbest - self.X) + c2 * r2 * (self.gbest - self.X)
            self.V = np.clip(self.V, -self.vmax, self.vmax)
            self.X = self.X + self.V

            # 边界与角度处理
            self.X = np.clip(self.X, self.lb, self.ub)
            self.X[:, 0] = np.array([angle_wrap(t) for t in self.X[:, 0]])

            # 评估与更新
            self.fitness = np.array([self.eval_fit(x) for x in self.X], dtype=float)
            improve_mask = self.fitness > self.pbest_fit
            self.pbest[improve_mask] = self.X[improve_mask]
            self.pbest_fit[improve_mask] = self.fitness[improve_mask]

            g_idx = int(np.argmax(self.pbest_fit))
            iter_best = float(np.max(self.fitness))
            if self.pbest_fit[g_idx] > self.gbest_fit:
                self.gbest = self.pbest[g_idx].copy()
                self.gbest_fit = float(self.pbest_fit[g_idx])

            # 每20次迭代做一次局部搜索与精英保留
            if it % 20 == 0:
                self.local_search()

            # 打印并更新收敛曲线
            print(f"[PSO {it}/{self.max_iters}] iter_best={iter_best:.6f}s, gbest={self.gbest_fit:.6f}s")
            self.it_hist.append(it)
            self.best_hist.append(self.gbest_fit)
            line.set_data(self.it_hist, self.best_hist)
            ax.relim(); ax.autoscale_view()
            fig.canvas.draw(); plt.pause(0.001)

        plt.ioff()
        plt.show()

        th, v, tr, dl = self.gbest
        return self.gbest_fit, float(th), float(v), float(tr), float(dl)

# 针对问题二的PSO类进行相同修改

def pso_optimize(swarm_size=48, max_iters=400, w_start=0.9, w_end=0.4, c1=1.8, c2=1.8, seed=None):
    rng = np.random.default_rng(seed)
    
    # 决策变量: [theta, vF, t_rel, dly]
    lb = np.array([-np.pi, 70.0, 0.0, 0.0], dtype=float)
    ub = np.array([ np.pi,140.0,10.0, 6.0], dtype=float)
    span = ub - lb

    # 初始化
    X = lb + rng.random((swarm_size, 4)) * span
    X[:, 0] = np.array([angle_wrap(t) for t in X[:, 0]])
    vmax = 0.2 * span
    V = rng.uniform(-vmax, vmax, size=(swarm_size, 4))

    # 适应度
    fitness = np.zeros(swarm_size, dtype=float)
    pbest = X.copy()
    pbest_fit = np.full(swarm_size, -np.inf, dtype=float)
    gbest = None
    gbest_fit = -np.inf

    # 在线收敛历史记录
    it_hist = [0]
    best_hist = [gbest_fit]
    print(f"[PSO 0/{max_iters}] iter_best={np.max(fitness):.6f}s, gbest={gbest_fit:.6f}s")
    
    # 迭代
    for it in range(1, max_iters + 1):
        w = w_start - (w_start - w_end) * it / max_iters
        c1 = 1.8 - (1.8 - 0.5) * it / max_iters
        c2 = 1.8 - (1.8 - 0.5) * it / max_iters
        r1 = rng.random((swarm_size, 4))
        r2 = rng.random((swarm_size, 4))

        # 速度与位置更新
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, -vmax, vmax)
        X = X + V

        # 边界与角度处理
        X = np.clip(X, lb, ub)
        X[:, 0] = np.array([angle_wrap(t) for t in X[:, 0]])

        # 评估与更新
        fitness = np.array([covered_time(x[2], x[3], x[1], x[0]) for x in X], dtype=float)
        improve_mask = fitness > pbest_fit
        pbest[improve_mask] = X[improve_mask]
        pbest_fit[improve_mask] = fitness[improve_mask]

        g_idx = int(np.argmax(pbest_fit))
        iter_best = float(np.max(fitness))
        if pbest_fit[g_idx] > gbest_fit:
            gbest = pbest[g_idx].copy()
            gbest_fit = float(pbest_fit[g_idx])

        # 仅记录历史，不绘图
        print(f"[PSO {it}/{max_iters}] iter_best={iter_best:.6f}s, gbest={gbest_fit:.6f}s")
        it_hist.append(it)
        best_hist.append(gbest_fit)
    
    # 最后绘制完整收敛曲线
    plot_convergence(it_hist, best_hist, "问题二PSO收敛曲线")
    
    th, v, tr, dl = gbest
    return gbest_fit, float(th), float(v), float(tr), float(dl)

def plot_convergence(it_hist, best_hist, title="PSO收敛曲线"):
    """绘制完整收敛曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("迭代次数", fontsize=12)
    ax.set_ylabel("遮蔽时间 (s)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.7)
    
    # 绘制收敛曲线
    ax.plot(it_hist, best_hist, "-o", ms=4, color="#FF7F0E", 
            linewidth=1.8, label="全局最优值")
    
    # 在最终点标记
    ax.scatter([it_hist[-1]], [best_hist[-1]], s=100, 
               marker="*", color="#C44E52", label=f"最终值: {best_hist[-1]:.4f}s")
    
    # 添加改进点标注
    improvements = np.diff(best_hist) > 0
    for i in range(1, len(it_hist)):
        if improvements[i-1] and i % max(1, len(it_hist)//10) == 0:
            ax.annotate(f"{best_hist[i]:.2f}", 
                       (it_hist[i], best_hist[i]),
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center',
                       fontsize=9,
                       arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("pso_q2_convergence_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

# 主优化流程
def main():
    t0 = time.time()
    optimizer = EnhancedPSO(
        swarm_size=50,      # 参考 a2.py 的规模
        max_iters=300,      # 上调迭代次数
        w_start=0.9, w_end=0.4,
        c1_start=2.5, c1_end=0.5,
        c2_start=0.5, c2_end=2.5,
        seed=42
    )
    score, theta_opt, vF_opt, t_rel_opt, dly_opt = optimizer.optimize()
    t_det_opt = t_rel_opt + dly_opt
    elapsed = time.time() - t0

    print(f"优化完成! 耗时: {elapsed:.2f}秒")
    print(f"最优航向角: {np.degrees(theta_opt):.4f}°")
    print(f"最优速度: {vF_opt:.4f} m/s")
    print(f"最优投放时刻: {t_rel_opt:.4f} s")
    print(f"最优起爆延迟: {dly_opt:.4f} s")
    print(f"起爆时刻: {t_det_opt:.4f} s")
    print(f"最大遮蔽时长(全视线): {score:.6f} s")

if __name__ == "__main__":
    main()