"""P2a-2 閉迴路 episode：MuJoCo 多機吊掛 + MPPI + 估計器（NOM/IND/CON/CEN/ORACLE）。

觀測（PLAN R1）：每機只有自己的 IMU 比力（由真實速度差分模擬，加白雜訊）、自己的位姿速度、自己的繩方向（加角度雜訊）、
payload 參考加速度、名義編隊幾何。估計器與控制器都不讀 payload 真實狀態、真實張力、真實 θ（ORACLE 除外，它就是真值）。
真值只用在：(1) 模擬感測器；(2) 評估指標。
"""
from __future__ import annotations

import time

import numpy as np

from smpc.adapt.estimators import Geometry, SharedEstimator, tension_model
from smpc.control.pd import quat_to_rot
from smpc.control.suspended_mppi import MPPIConfig, SuspendedMPPITeam
from smpc.control.trajectories import Traj
from smpc.eval.episode import DURATION, M_NOM, make_env, rope_dirs

SIGMA_A = 0.15        # 加速度計白雜訊 (m/s^2 / 樣本 @100 Hz)：MEMS 本體約 0.04，加機體振動取 0.15（同 P0.5 A10，假設非實測）
SIGMA_Q = 0.005       # 繩方向角度雜訊 (rad, 每軸)：約 0.3 度（同 P0.5 A10，假設）
SLACK_DEPTH = 0.10    # m
SLACK_TMIN = 0.3      # N
ETA_FAULT = 0.75
DM_RANGE = (-0.3, 0.4)
DC_MAX = 0.04
TH_CLIP = (np.array([-0.7, -0.15, -0.15]), np.array([1.0, 0.15, 0.15]))   # 只在 θ̂ -> 控制器 的介面夾（估計器內部不夾）


def draw_truth(seed, eta_fault=False, n=3):
    rng = np.random.default_rng([seed, 1])
    dm = rng.uniform(*DM_RANGE)
    r = DC_MAX * np.sqrt(rng.uniform())
    ph = rng.uniform(0, 2 * np.pi)
    th = np.array([dm, r * np.cos(ph), r * np.sin(ph)])
    eta = np.ones(n)
    k = int(rng.integers(0, n))
    if eta_fault:
        eta[k] = ETA_FAULT
    return th, eta, k


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def run_episode_est(method, traj_name, seed, graph="path", iters=2, noise=True, eta_fault=False, n=3,
                    cfg: MPPIConfig | None = None, T=None, record=False, lam=0.9995, relin=True, prior_std=None, dc_off=False):
    """method: NOM | IND | CON | CEN | ORACLE。回傳 metrics dict（record=True 另含 'series'）。"""
    th_true, eta_true, k_fault = draw_truth(seed, eta_fault, n)
    init_rng = np.random.default_rng([seed, 2])
    sens = np.random.default_rng([seed, 3])
    cfg = cfg or MPPIConfig.load("configs/mppi_frozen.json")
    mass, com = M_NOM + th_true[0], (th_true[1], th_true[2], 0.0)
    env = make_env(n, mass, com)
    traj = Traj(traj_name)
    T = T or DURATION[traj_name]
    p0 = traj.ref(0.0)[0][0] + init_rng.normal(0, [0.03, 0.03, 0.02])
    o = env.reset(payload_pos=p0)
    env.eta = np.tile(eta_true[:, None], (1, 4))     # reset() 會把 eta 設回 1，所以必須在 reset 之後設
    env.kick_payload(init_rng.normal(0, 0.1, 3))
    th0 = th_true if method == "ORACLE" else np.zeros(3)
    team = SuspendedMPPITeam(env, traj, cfg, theta_hat=th0, m_nom=M_NOM, seed=seed)
    geom = Geometry.from_env(env, cfg.cone_deg, M_NOM, cfg.w_moment)
    est = SharedEstimator(method, geom, lam=lam, sigma_f=SIGMA_A * env.p.mass, graph=graph, iters=iters, relin=relin,
                          theta_true=th_true, prior_std=prior_std)
    dt, m_q, g = env.dt, env.p.mass, env.p.g
    ez = np.array([0, 0, g])
    sa, sq = (SIGMA_A, SIGMA_Q) if noise else (0.0, 0.0)

    def meas_dir():
        return _unit(rope_dirs(env) + sq * sens.standard_normal((n, 3)))

    q_next = meas_dir()
    steps = int(round(T / dt))
    P, R, tilt, swing, tr_T, ts_time, est_time = [], [], [], [], [], [], []
    th_hist, nees_hist, ze_hist, clipn, inactive = [], [], [], 0, 0
    rec = {k: [] for k in ("y", "y_clean", "f", "q", "q_true", "a_ref", "active", "tilt", "T_true", "tm")} if record else None
    slack_events, prev, finite = 0, False, True
    for k in range(steps):
        t = k * dt
        qs0 = o["quad_states"]
        q_now = q_next
        U = team.act(t, qs0, q_now)
        Uc = np.clip(U, 0.0, env.p.fmax)
        o = env.step(U)
        qs1, ps = o["quad_states"], o["payload_state"]
        if not (np.isfinite(qs1).all() and np.isfinite(ps).all()):
            finite = False
            break
        q_next = meas_dir()
        # ---- 模擬感測器 ----
        a_spec = (qs1[:, 7:10] - qs0[:, 7:10]) / dt + ez                       # 區間平均比力（世界座標）
        a_imu = a_spec + sa * sens.standard_normal((n, 3))
        b3 = _unit(np.array([quat_to_rot(qs0[i, 3:7])[:, 2] + quat_to_rot(qs1[i, 3:7])[:, 2] for i in range(n)]))
        f = Uc.sum(1)[:, None] * b3                                              # 自己送出的推力向量（指令；不含 η）
        q_est = _unit(q_now + q_next)
        y = m_q * a_imu - f
        tm = t + 0.5 * dt
        a_ref = traj.ref(tm)[2][0]
        # slack（幾何）：自己的位置到「參考掛點」的距離 < L - margin -> 資訊權重 0
        pq_mid = 0.5 * (qs0[:, :3] + qs1[:, :3])
        pa_ref = traj.ref(tm)[0][0][None] + env.attach_body
        # 控制器用的 3 cm 邊界在這裡不能用：距離是對「參考」掛點量的，payload 追蹤偏差（NOM 約 4-7 cm）會把 16-94% 的樣本誤判為 slack
        # （預試 seeds 2000-2002 實測）。改為：深度內縮 > SLACK_DEPTH，或名義模型（θ=0）預測張力 < SLACK_TMIN（只用參考加速度 + 幾何）。
        T_pred = np.array([tension_model(geom, np.zeros(3), a_ref, i)[0] for i in range(n)])   # 用名義 θ=0：不可依賴 θ̂，否則 θ̂ 壞掉 -> 全判 slack -> 估計器凍結（v1 實測鎖死）
        active = (np.linalg.norm(pq_mid - pa_ref, axis=1) >= env.L - SLACK_DEPTH) & (T_pred >= SLACK_TMIN)
        inactive += int((~active).sum())
        t0 = time.thread_time()
        est.update(a_ref, q_est, f, y, active)
        est_time.append(time.thread_time() - t0)
        if method in ("IND", "CON", "CEN"):
            for i in range(n):
                th_c = np.clip(est.theta[i], *TH_CLIP)
                if dc_off:
                    th_c = th_c * np.array([1.0, 0.0, 0.0])      # 診斷：只把 δm 交給控制器
                clipn += int(np.any(th_c != est.theta[i]))
                team.set_theta(i, th_c)
        # ---- 指標 ----
        th_hist.append(est.theta.copy())
        nees_hist.append(est.nees(th_true))
        ze_hist.append(est.zeta.copy())
        P.append(ps[:3]); R.append(traj.ref(t + dt)[0][0])
        ts_time.append(team.step_time.copy())
        w_, x_, y_, z_ = ps[3:7]
        tilt.append(np.degrees(np.arccos(np.clip(1 - 2 * (x_ * x_ + y_ * y_), -1, 1))))
        cen = qs1[:, :3].mean(0) - ps[:3]
        swing.append(np.degrees(np.arctan2(np.linalg.norm(cen[:2]), cen[2])))
        sl = bool((o["tension_truth"] < 0.02).any())
        slack_events += int(sl and not prev)
        prev = sl
        tr_T.append(o["tension_truth"].copy())
        if record:
            qt = _unit(rope_dirs(env))
            rec["y"].append(y); rec["y_clean"].append(m_q * a_spec - f); rec["f"].append(f); rec["q"].append(q_est)
            rec["q_true"].append(qt); rec["a_ref"].append(a_ref); rec["active"].append(active)
            rec["tilt"].append(tilt[-1]); rec["T_true"].append(o["tension_truth"].copy()); rec["tm"].append(tm)
    P, R = np.array(P), np.array(R)
    err = np.linalg.norm(P - R, axis=1) if len(P) else np.array([np.inf])
    n_last = int(1.0 / dt)
    final_err = float(err[-n_last:].mean()) if finite else np.inf
    ok = bool(finite and final_err < 0.10 and err.max() < 0.5)
    m = dict(method=method + (f"-{graph}" if method == "CON" else ""), traj=traj_name, seed=seed, graph=graph,
             noise=noise, eta_fault=eta_fault, prior_std=None if prior_std is None else list(prior_std), dc_off=dc_off, theta_true=th_true.tolist(), eta_true=eta_true.tolist(),
             rmse=float(np.sqrt((err ** 2).mean())) if finite else np.inf, max_err=float(err.max()),
             final_err=final_err, max_tilt_deg=float(np.max(tilt)) if tilt else np.inf,
             max_swing_deg=float(np.max(swing)) if swing else np.inf, slack_events=int(slack_events),
             success=ok, finite=finite, clip_frac=clipn / max(1, steps * n), inactive_frac=inactive / max(1, steps * n),
             t_est_ms=float(np.mean(est_time) * 1e3) if est_time else 0.0,
             t_mppi_ms=float(np.mean(ts_time) * 1e3) if ts_time else 0.0)
    if th_hist:
        th_h, nees_h, ze_h = np.array(th_hist), np.array(nees_hist), np.array(ze_hist)     # (steps,n,3),(steps,n),(steps,n)
        e = th_h - th_true
        for tt in (2.0, 5.0, 10.0):
            ix = min(int(round(tt / dt)) - 1, len(e) - 1)
            m[f"dm_err@{tt:g}"] = float(np.abs(e[ix, :, 0]).mean())
            m[f"dc_err@{tt:g}"] = float(np.linalg.norm(e[ix, :, 1:], axis=1).mean())
            m[f"nees@{tt:g}"] = float(nees_h[ix].mean())
        m["eta_err@5"] = float(np.abs(ze_h[min(499, len(e) - 1), k_fault] - (eta_true[k_fault] - 1)))
        m["theta_series"] = th_h[::10].mean(1).tolist()    # 0.1 s 取樣，各機平均 θ̂（作圖用）
    if record:
        m["series"] = {k: np.array(v) for k, v in rec.items()}
    return m
