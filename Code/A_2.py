# 问题二优化框架（粗网格搜索+局部细化）
import numpy as np
import time
import matplotlib.pyplot as plt  # 新增：在线可视化

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 常量定义（同问题一）
T = np.array([0.0, 200.0, 0.0])  # 真目标位置

# === 目标轴线采样（新增）===
H_target = 10.0                   # 上下底面圆心的垂直间距，可按实际调整
N_T_samples = 10                  # 轴线上采样点数
T_bottom = T
T_top = T_bottom + np.array([0.0, 0.0, H_target])
T_arr = np.linspace(T_bottom, T_top, N_T_samples)  # 形状 (N_T_samples, 3)

M0 = np.array([20000.0, 0.0, 2000.0])  # 导弹初始位置
F0 = np.array([17800.0, 0.0, 1800.0])  # 无人机初始位置
vM = 300.0  # 导弹速度
g = 9.8  # 重力加速度
v_sink = 3.0  # 云团下沉速度
r_cloud = 10.0  # 云团有效半径
t_window = 20.0  # 云团有效时间

# 离散时间步长
DT = 0.02  # 秒

def covered_time(t_rel, dly, vF, theta):
    """计算给定参数下(轴线10点采样，全视线判定)的遮蔽时长"""
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

    # 计算每条射线相对云团中心的最小距离平方数组，长度 N_T_samples
    def d2_each(t):
        C_t = C(t)
        if np.isnan(C_t).any():
            return np.full(N_T_samples, np.inf, dtype=float)
        M_t = M(t)

        # 向量化计算到线段 T_i -- M_t 的最短距离平方
        TM = M_t - T_arr                         # (N,3)
        den = np.einsum('ij,ij->i', TM, TM)      # (N,)
        CT = C_t - T_arr                         # (N,3)

        # 避免除零：den==0 视为 s=0（最近点为 T_i）
        s = np.zeros_like(den, dtype=float)
        mask = den > 0.0
        s[mask] = np.einsum('ij,ij->i', CT[mask], TM[mask]) / den[mask]
        s = np.clip(s, 0.0, 1.0)

        closest = T_arr + (s[:, None] * TM)      # (N,3)
        diff = closest - C_t
        d2 = np.einsum('ij,ij->i', diff, diff)   # (N,)
        return d2

    # 时间采样并按“全视线覆盖”判定
    ts = np.arange(t_start, t_end, DT)
    thr = r_cloud**2 + 1e-12
    covered_flags = np.fromiter(
        (np.all(d2_each(t) <= thr) for t in ts),
        dtype=bool, count=ts.size
    )
    return float(np.sum(covered_flags) * DT)

def refine(th, v, tr, dl, iters=300, step=(0.1, 1.0, 0.05, 0.05)):
    """
    在粗搜的较优解附近做随机邻域搜索并逐步缩小步长
    参数: th, v, tr, dl: 初始解(来自粗搜)
          iters: 迭代次数
          step: 初始扰动尺度(Δtheta, Δv, Δt_rel, Δdl)
    返回: (score, theta, vF, t_rel, dly)
    """
    sth, sv, sr, sd = step
    cur = (covered_time(tr, dl, v, th), th, v, tr, dl)

    # 在线收敛曲线：best-so-far vs iteration
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("遮掩时间 vs 迭代次数（best-so-far）")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("遮掩时间 (s)")
    ax.grid(True, linestyle=":")
    line, = ax.plot([], [], "-o", ms=3)
    it_hist = [0]
    best_hist = [cur[0]]
    line.set_data(it_hist, best_hist)
    ax.relim(); ax.autoscale_view()
    fig.canvas.draw(); fig.canvas.flush_events()
    print(f"[refine 0/{iters}] cand={cur[0]:.6f}s, best={cur[0]:.6f}s")

    for k in range(iters):
        th2 = cur[1] + np.random.uniform(-sth, sth)
        v2 = np.clip(cur[2] + np.random.uniform(-sv, sv), 70, 140)
        tr2 = max(0.0, cur[3] + np.random.uniform(-sr, sr))
        dl2 = max(0.0, cur[4] + np.random.uniform(-sd, sd))

        val = covered_time(tr2, dl2, v2, th2)
        # 打印本次候选与当前最优
        print(f"[refine {k+1}/{iters}] cand={val:.6f}s, best={cur[0]:.6f}s", end="")

        improved = False
        if val > cur[0]:
            cur = (val, th2, v2, tr2, dl2)
            improved = True
            print("  -> improved")
        else:
            print("")

        # 记录 best-so-far 并更新曲线
        it_hist.append(k + 1)
        best_hist.append(cur[0])
        line.set_data(it_hist, best_hist)
        ax.relim(); ax.autoscale_view()
        fig.canvas.draw(); plt.pause(0.001)

        if (k + 1) % 40 == 0:
            sth *= 0.6; sv *= 0.6; sr *= 0.6; sd *= 0.6

    plt.ioff()
    plt.show()  # 最终停留显示收敛曲线
    return cur

# 主优化流程
def main():
    # 粗网格搜索
    t0 = time.time()

    # 参数范围
    thetas = np.linspace(-np.pi, np.pi, 15)   # 航向角
    vFs = np.linspace(70, 140, 10)            # 速度
    t_rels = np.linspace(0, 10, 11)           # 投放时间
    dlays = np.linspace(0, 6, 7)              # 起爆延迟

    best = (-np.inf, 0, 0, 0, 0)  # (score, theta, vF, t_rel, dly)

    # 网格搜索
    for theta in thetas:
        for vF in vFs:
            for t_rel in t_rels:
                for dly in dlays:
                    score = covered_time(t_rel, dly, vF, theta)
                    if score > best[0]:
                        best = (score, theta, vF, t_rel, dly)

    # 局部细化
    best = refine(*best[1:], iters=240)

    elapsed = time.time() - t0

    # 解包最优解
    score, theta_opt, vF_opt, t_rel_opt, dly_opt = best
    t_det_opt = t_rel_opt + dly_opt

    print(f"优化完成! 耗时: {elapsed:.2f}秒")
    print(f"最优航向角: {np.degrees(theta_opt):.4f}°")
    print(f"最优速度: {vF_opt:.4f} m/s")
    print(f"最优投放时刻: {t_rel_opt:.4f} s")
    print(f"最优起爆延迟: {dly_opt:.4f} s")
    print(f"起爆时刻: {t_det_opt:.4f} s")
    print(f"最大遮蔽时长(全视线): {score:.6f} s")

if __name__ == "__main__":
    main()