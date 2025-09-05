# 问题二优化框架（粗网格搜索+局部细化）
import numpy as np
import time

# 常量定义（同问题一）
T = np.array([0.0, 200.0, 0.0])  # 真目标位置
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
    """计算给定参数下的遮蔽时长"""
    t_det = t_rel + dly
    t_start = t_det
    t_end = t_det + t_window
    
    # 无人机方向向量
    uF = np.array([np.cos(theta), np.sin(theta), 0])
    
    # 无人机位置函数
    def F(t):
        return F0 + vF * t * uF
    
    # 干扰弹位置函数
    def B(t):
        if t < t_rel:
            return F(t)
        dt = t - t_rel
        rel_pos = F(t_rel)
        return rel_pos + vF * uF * dt + np.array([0, 0, -0.5*g*dt**2])
    
    # 云团中心位置函数
    C_det = B(t_det)
    def C(t):
        if t < t_det:
            return np.array([np.nan, np.nan, np.nan])
        dt = t - t_det
        return C_det + np.array([0, 0, -v_sink*dt])
    
    # 导弹位置函数
    uM = -M0 / np.linalg.norm(M0)
    def M(t):
        return M0 + vM * t * uM
    
    # 计算最小距离平方函数
    def d2(t):
        C_t = C(t)
        M_t = M(t)
        if any(np.isnan(C_t)):
            return np.inf
        TM = M_t - T
        CT = C_t - T
        s = np.dot(CT, TM) / np.dot(TM, TM)
        s_clamped = max(0.0, min(1.0, s))
        closest = T + s_clamped * TM
        return np.sum((closest - C_t)**2)
    
    # 时间采样
    ts = np.arange(t_start, t_end, DT)
    d2_vals = np.array([d2(t) for t in ts])
    covered = d2_vals <= (r_cloud**2 + 1e-12)
    return np.sum(covered) * DT

def refine(th, v, tr, dl, iters=240, step=(0.1, 1.0, 0.05, 0.05)):
    """
    在粗搜的较优解附近做随机邻域搜索并逐步缩小步长
    参数: th, v, tr, dl: 初始解(来自粗搜)
          iters: 迭代次数
          step: 初始扰动尺度(Δtheta, Δv, Δt_rel, Δdl)
    策略:
      - 每次随机扰动四个维度(带边界裁剪)
      - 若新解更优则接受
      - 每40次缩小一次扰动步长(近似"降温")
    返回: (score, theta, vF, t_rel, dly)
    """
    sth, sv, sr, sd = step
    cur = (covered_time(tr, dl, v, th), th, v, tr, dl)
    
    for k in range(iters):
        # 随机扰动
        th2 = cur[1] + np.random.uniform(-sth, sth)
        v2 = np.clip(cur[2] + np.random.uniform(-sv, sv), 70, 140)
        tr2 = max(0.0, cur[3] + np.random.uniform(-sr, sr))
        dl2 = max(0.0, cur[4] + np.random.uniform(-sd, sd))
        
        # 计算新解
        val = covered_time(tr2, dl2, v2, th2)
        
        # 接受更优解
        if val > cur[0]:
            cur = (val, th2, v2, tr2, dl2)
        
        # 每40次迭代缩小步长
        if (k+1) % 40 == 0:
            sth *= 0.6
            sv *= 0.6
            sr *= 0.6
            sd *= 0.6
    
    return cur

# 主优化流程
def main():
    # 粗网格搜索
    t0 = time.time()
    
    # 参数范围
    thetas = np.linspace(-np.pi, np.pi, 15)  # 航向角
    vFs = np.linspace(70, 140, 10)          # 速度
    t_rels = np.linspace(0, 10, 11)          # 投放时间
    dlays = np.linspace(0, 6, 7)             # 起爆延迟
    
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
    print(f"最大遮蔽时长: {score:.6f} s")
    
    # 这里可以添加详细的时间区间计算和可视化代码

if __name__ == "__main__":
    main()