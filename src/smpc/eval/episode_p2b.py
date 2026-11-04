"""P2b 閉迴路 episode：在 P2a-2 的 episode_est 上加入
  - 軌跡速度倍率 speed
  - 本地 η 補償（eta_comp: "none" | "oracle" | "est"）
  - 不通訊基線：IND-DM（IND-δm）、IND-TIGHT（δm 先驗 0.25）、DOB（DOB-hold）、LS（LOADSHARE）、LSN（LOADSHARE-N）
  - CEN-ALL（不估本地 ζ，η 全被當共享量吸收）
  - 每機「張力預測誤差」（只評估用）：每步控制器對 0.5 s 後張力的預測 vs 該時刻真實張力
觀測模型與雜訊與 P2a-2 完全相同（IMU 白雜訊 0.15、繩方向 0.005 rad），估計器與控制器都不讀 payload 真實狀態／張力／θ（ORACLE 除外）。
"""
from __future__ import annotations

import time

import numpy as np

from smpc.adapt.estimators import Geometry, SharedEstimator, tension_model
from smpc.adapt.local_baselines import DOBHold, LoadShare, LocalDm
from smpc.control.pd import quat_to_rot
from smpc.control.suspended_mppi import MPPIConfig, SuspendedMPPITeam
from smpc.control.trajectories import Traj
from smpc.eval.episode import DURATION, M_NOM, make_env, rope_dirs
from smpc.eval.episode_est import (SIGMA_A, SIGMA_Q, SLACK_DEPTH, SLACK_TMIN, TH_CLIP, ETA_FAULT, _unit, draw_truth)

SHARED = ("IND", "CON", "CEN", "CEN-ALL", "CEN-ATT")
LOCAL_EST = ("IND-DM", "IND-TIGHT")
FORCE_BASED = ("DOB", "LS", "LSN")
ALL_METHODS = ("NOM", "ORACLE") + SHARED + LOCAL_EST + FORCE_BASED
LEAD = 10        # 0.5 s = 10 個規劃步（dt_plan 0.05）


def run_episode_p2b(method, traj_name, seed, speed=1.0, graph="path", iters=2, noise=True, eta_fault=False, eta_comp="none",
                    n=3, cfg: MPPIConfig | None = None, T=None, record=False, lam=0.9995, fc=2.0, dc_off=False,
                    eta_true_override=None, th_true_override=None):
    """eta_comp: none（不補償）| oracle（η 真值）| est（本地 ζ̂ -> η̂ = 1+ζ̂，需要估計器有 ζ）。"""
    assert method in ALL_METHODS
    th_true, eta_true, k_fault = draw_truth(seed, eta_fault, n)
    if th_true_override is not None:
        th_true = np.asarray(th_true_override, float)
    if eta_true_override is not None:
        eta_true = np.asarray(eta_true_override, float)
    init_rng = np.random.default_rng([seed, 2])
    sens = np.random.default_rng([seed, 3])
    cfg = cfg or MPPIConfig.load("configs/mppi_frozen.json")
    mass, com = M_NOM + th_true[0], (th_true[1], th_true[2], 0.0)
    env = make_env(n, mass, com)
    traj = Traj(traj_name, speed=speed)
    T = T or DURATION[traj_name]
    p0 = traj.ref(0.0)[0][0] + init_rng.normal(0, [0.03, 0.03, 0.02])
    o = env.reset(payload_pos=p0)
    env.eta = np.tile(eta_true[:, None], (1, 4))
    env.kick_payload(init_rng.normal(0, 0.1, 3))
    th0 = th_true if method == "ORACLE" else np.zeros(3)
    team = SuspendedMPPITeam(env, traj, cfg, theta_hat=th0, m_nom=M_NOM, seed=seed)
    geom = Geometry.from_env(env, cfg.cone_deg, M_NOM, cfg.w_moment)
    dt, m_q, g = env.dt, env.p.mass, env.p.g
    sigma_f = SIGMA_A * m_q
    ez = np.array([0, 0, g])
    est = loc = dob = ls = None
    if method in ("NOM", "ORACLE", "IND", "CON", "CEN"):
        est = SharedEstimator(method, geom, lam=lam, sigma_f=sigma_f, graph=graph, iters=iters, theta_true=th_true)
    elif method == "CEN-ATT":   # P2b-B V1：payload 姿態感知（CEN；3 機 path 圖 2 輪 flooding 等價）
        from smpc.adapt.attitude import ReplayCEN
        att = ReplayCEN(geom, env.L, "att", lam=lam, sigma_f=sigma_f, sigma_q=SIGMA_Q)
        est = att
    elif method == "CEN-ALL":
        est = SharedEstimator("CEN", geom, lam=lam, sigma_f=sigma_f, zeta_std=1e-4)
    elif method in LOCAL_EST:
        loc = LocalDm(geom, prior_dm=0.25 if method == "IND-TIGHT" else 0.5, lam=lam, sigma_f=sigma_f)
    elif method == "DOB":
        dob = DOBHold(n, fc, dt)
        for i in range(n):
            team.set_tension_fn(i, dob.force_fn(i))
    elif method in ("LS", "LSN"):
        ls = LoadShare(n, fc, dt, M_NOM, g, geom, nominal=(method == "LSN"))
        for i in range(n):
            team.set_tension_fn(i, ls.force_fn(i))
    if eta_comp == "oracle":
        for i in range(n):
            team.set_eta(i, eta_true[i])
    sa, sq = (SIGMA_A, SIGMA_Q) if noise else (0.0, 0.0)

    def meas_dir():
        return _unit(rope_dirs(env) + sq * sens.standard_normal((n, 3)))

    q_next = meas_dir()
    steps = int(round(T / dt))
    P, R, tilt, swing, tr_T, pred10, pred0 = [], [], [], [], [], [], []
    th_hist, ze_hist, eta_hist, clipn, inactive = [], [], [], 0, 0
    rec = {k: [] for k in ("y", "f", "q", "a_ref", "active", "tilt", "T_true", "pq", "tm")} if record else None
    slack_events, prev, finite = 0, False, True
    for k in range(steps):
        t = k * dt
        qs0 = o["quad_states"]
        q_now = q_next
        U = team.act(t, qs0, q_now)
        pred10.append([a.last_T[LEAD] for a in team.agents]); pred0.append([a.last_T[0] for a in team.agents])
        Uc = np.clip(U, 0.0, env.p.fmax)
        o = env.step(U)
        qs1, ps = o["quad_states"], o["payload_state"]
        if not (np.isfinite(qs1).all() and np.isfinite(ps).all()):
            finite = False
            break
        q_next = meas_dir()
        a_spec = (qs1[:, 7:10] - qs0[:, 7:10]) / dt + ez
        a_imu = a_spec + sa * sens.standard_normal((n, 3))
        b3 = _unit(np.array([quat_to_rot(qs0[i, 3:7])[:, 2] + quat_to_rot(qs1[i, 3:7])[:, 2] for i in range(n)]))
        f = Uc.sum(1)[:, None] * b3
        q_est = _unit(q_now + q_next)
        y = m_q * a_imu - f
        tm = t + 0.5 * dt
        a_ref = traj.ref(tm)[2][0]
        pq_mid = 0.5 * (qs0[:, :3] + qs1[:, :3])
        pa_ref = traj.ref(tm)[0][0][None] + env.attach_body
        T_pred = np.array([tension_model(geom, np.zeros(3), a_ref, i)[0] for i in range(n)])
        active = (np.linalg.norm(pq_mid - pa_ref, axis=1) >= env.L - SLACK_DEPTH) & (T_pred >= SLACK_TMIN)
        inactive += int((~active).sum())
        if method == "CEN-ATT":
            est.update(a_ref, q_est, f, y, active, pq_mid)
            theta_now, zeta_now = np.tile(est.theta, (n, 1)), est.zeta
        elif est is not None:
            est.update(a_ref, q_est, f, y, active)
            theta_now, zeta_now = est.theta, est.zeta
        elif loc is not None:
            loc.update(a_ref, q_est, f, y, active)
            theta_now, zeta_now = loc.theta, loc.zeta
        else:
            theta_now = zeta_now = None
        if dob is not None:
            dob.update(y, active)
        if ls is not None:
            ls.update(a_ref, q_est, y, active)
        if method in SHARED or method in LOCAL_EST:
            for i in range(n):
                th_c = np.clip(theta_now[i], *TH_CLIP)
                if dc_off:
                    th_c = th_c * np.array([1.0, 0.0, 0.0])
                clipn += int(np.any(th_c != theta_now[i]))
                team.set_theta(i, th_c)
        if eta_comp == "est" and zeta_now is not None:
            for i in range(n):
                team.set_eta(i, 1.0 + zeta_now[i])
        if theta_now is not None:
            th_hist.append(theta_now.copy()); ze_hist.append(zeta_now.copy())
        eta_hist.append([a.eta_hat for a in team.agents])
        P.append(ps[:3]); R.append(traj.ref(t + dt)[0][0])
        w_, x_, y_, z_ = ps[3:7]
        tilt.append(np.degrees(np.arccos(np.clip(1 - 2 * (x_ * x_ + y_ * y_), -1, 1))))
        cen = qs1[:, :3].mean(0) - ps[:3]
        swing.append(np.degrees(np.arctan2(np.linalg.norm(cen[:2]), cen[2])))
        sl = bool((o["tension_truth"] < 0.02).any())
        slack_events += int(sl and not prev)
        prev = sl
        tr_T.append(o["tension_truth"].copy())
        if record:
            rec["y"].append(y); rec["f"].append(f); rec["q"].append(q_est); rec["a_ref"].append(a_ref)
            rec["active"].append(active); rec["tilt"].append(tilt[-1]); rec["T_true"].append(o["tension_truth"].copy())
            rec["pq"].append(pq_mid); rec["tm"].append(tm)
    P, R = np.array(P), np.array(R)
    err = np.linalg.norm(P - R, axis=1) if len(P) else np.array([np.inf])
    n_last = int(1.0 / dt)
    final_err = float(err[-n_last:].mean()) if finite else np.inf
    ok = bool(finite and final_err < 0.10 and err.max() < 0.5)
    m = dict(method=method, traj=traj_name, speed=speed, seed=seed, eta_fault=eta_fault, eta_comp=eta_comp, fc=fc,
             theta_true=th_true.tolist(), eta_true=eta_true.tolist(), k_fault=int(k_fault),
             rmse=float(np.sqrt((err ** 2).mean())) if finite else np.inf, max_err=float(err.max()), final_err=final_err,
             max_tilt_deg=float(np.max(tilt)) if tilt else np.inf, max_swing_deg=float(np.max(swing)) if swing else np.inf,
             slack_events=int(slack_events), success=ok, finite=finite, clip_frac=clipn / max(1, steps * n),
             inactive_frac=inactive / max(1, steps * n))
    # ---- 張力預測誤差（只評估用）：t >= 2 s，且 t+0.5 s 仍在 episode 內 ----
    if finite and len(tr_T) > 300:
        TT = np.array(tr_T)                      # (steps,n)，TT[k] = 步 k 之後的張力
        p10, p0 = np.array(pred10), np.array(pred0)
        lead = 50
        ks = np.arange(200, len(TT) - lead)
        e10 = p10[ks] - TT[ks + lead - 1]
        e0 = p0[ks] - TT[ks]
        m["tpe10"] = float(np.sqrt((e10 ** 2).mean()))            # N，所有機合併 RMSE
        m["tpe0"] = float(np.sqrt((e0 ** 2).mean()))
        m["tension_rms"] = float(np.sqrt((TT[ks] ** 2).mean()))
    if th_hist:
        th_h, ze_h = np.array(th_hist), np.array(ze_hist)
        e = th_h - th_true
        for tt in (2.0, 5.0, 10.0):
            ix = min(int(round(tt / dt)) - 1, len(e) - 1)
            m[f"dm_err@{tt:g}"] = float(np.abs(e[ix, :, 0]).mean())
            m[f"dc_err@{tt:g}"] = float(np.linalg.norm(e[ix, :, 1:], axis=1).mean())
        ix = min(499, len(e) - 1)
        m["zeta_err@5"] = float(np.abs(ze_h[ix] - (eta_true - 1)).mean())
        m["zeta_err_fault@5"] = float(np.abs(ze_h[ix, k_fault] - (eta_true[k_fault] - 1)))
    m["eta_hat_final"] = eta_hist[-1] if eta_hist else None
    if record:
        m["series"] = {k: np.array(v) for k, v in rec.items()}
    return m
