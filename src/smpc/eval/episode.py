"""P2a 的 episode 執行器與指標。控制器可為 MPPI team 或 PD 基線；真值只用在評估指標與（模擬的）繩方向感測。"""
from __future__ import annotations

import time

import numpy as np

from smpc.control.suspended_hover import SuspendedHover
from smpc.control.suspended_mppi import MPPIConfig, SuspendedMPPITeam
from smpc.control.trajectories import Traj
from smpc.envs.multi_payload import MultiPayloadEnv

DURATION = {"hover": 6.0, "p2p": 8.0, "circle": 10.0, "fig8": 10.0}
M_NOM = 1.0


def rope_dirs(env):
    pa, _, pq, _, _ = env._rope_kin()
    d = pa - pq
    return d / np.linalg.norm(d, axis=1, keepdims=True)


class PDStatic:
    """既有 SuspendedHover：PD + 已知 payload 的靜力前饋，只吃位置參考。"""
    name = "pd_static"

    def __init__(self, env, traj, mass, com):
        self.env, self.traj = env, traj
        self.h = SuspendedHover(env)
        self.h.set_known_payload(mass, com)
        self.off = env.hover_positions(np.zeros(3))
        self.step_time = np.zeros(env.n)

    def act(self, t, qs, rd=None):
        p = self.traj.ref(t)[0][0]
        return self.h(qs, self.off + p)


class PDDynFF:
    """PD + 與 MPPI 相同的動態前饋（a_ref 與由 θ̂_s 預測的繩力），用同一份 team 內的前饋計算。"""
    name = "pd_dynff"

    def __init__(self, team):
        self.team = team
        self.step_time = np.zeros(team.env.n)

    def act(self, t, qs, rd=None):
        out = []
        for i, a in enumerate(self.team.agents):
            pq, vq, d, a_ff, pa, _ = a._feedforward(t, None if rd is None else rd[i])
            # acc_d = 推力加速度前饋 a_ff[0]；PD 項用位置／速度誤差
            out.append(a.pd(qs[i], pq[0], vq[0], acc_d=a_ff[0]))
        return np.stack(out)


PAYLOAD_RADIUS = 0.15
ATTACH_Z = 0.1


def make_env(n=3, mass=1.0, com=(0, 0, 0), radius=None):
    return MultiPayloadEnv(n=n, payload_mass=mass, payload_com=com, payload_radius=radius or PAYLOAD_RADIUS, attach_z=ATTACH_Z)


def run_episode(method, traj_name, seed, com=(0.0, 0.0, 0.0), mass=1.0, n=3, cfg: MPPIConfig | None = None,
                T=None, theta_hat="oracle", record=False, perturb=True):
    """method: 'mppi' | 'pd_static' | 'pd_dynff'。theta_hat: 'oracle' | 'nominal' | 長度 3 陣列。"""
    rng = np.random.default_rng(seed)
    env = make_env(n, mass, com)
    traj = Traj(traj_name)
    T = T or DURATION[traj_name]
    p0 = traj.ref(0.0)[0][0] + (rng.normal(0, [0.03, 0.03, 0.02]) if perturb else 0)
    o = env.reset(payload_pos=p0)
    if perturb:
        env.kick_payload(rng.normal(0, 0.1, 3))
    com3 = np.asarray(com, float)
    th = {"oracle": np.array([mass - M_NOM, com3[0], com3[1]]), "nominal": np.zeros(3)}.get(theta_hat, theta_hat) \
        if isinstance(theta_hat, str) else np.asarray(theta_hat, float)
    if method == "pd_static":
        ctrl = PDStatic(env, traj, mass, com)
    else:
        cfg = cfg or MPPIConfig()
        team = SuspendedMPPITeam(env, traj, cfg, theta_hat=th, m_nom=M_NOM, seed=seed)
        ctrl = team if method == "mppi" else PDDynFF(team)
    steps = int(round(T / env.dt))
    P, R, Tn, ts_time, tilt, swing, slack_flag = [], [], [], [], [], [], []
    ff_T, tr_T = [], []
    slack_events, prev = 0, False
    finite = True
    for k in range(steps):
        t = k * env.dt
        U = ctrl.act(t, o["quad_states"], rope_dirs(env))
        o = env.step(U)
        ps = o["payload_state"]
        if not (np.isfinite(o["quad_states"]).all() and np.isfinite(ps).all()):
            finite = False
            break
        pref = traj.ref(t + env.dt)[0][0]
        P.append(ps[:3]); R.append(pref)
        ts_time.append(np.asarray(ctrl.step_time).copy())
        w, x, y, z = ps[3:7]
        tilt.append(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))
        cen = o["quad_states"][:, :3].mean(0) - ps[:3]
        swing.append(np.degrees(np.arctan2(np.linalg.norm(cen[:2]), cen[2])))
        sl = bool((o["tension_truth"] < 0.02).any())
        slack_events += int(sl and not prev)
        prev = sl
        tr_T.append(o["tension_truth"].copy())
        if method != "pd_static":
            ff_T.append(ctrl.feedforward_tension() if method == "mppi" else ctrl.team.feedforward_tension())
    P, R = np.array(P), np.array(R)
    err = np.linalg.norm(P - R, axis=1) if len(P) else np.array([np.inf])
    ff_rel = np.nan
    if ff_T and finite:   # 繩力前饋（θ̂ 與參考預測）對真值張力的相對誤差，診斷用，控制器看不到真值
        a_, b_ = np.array(ff_T), np.array(tr_T)
        ff_rel = float(np.sqrt(((a_ - b_) ** 2).mean()) / np.sqrt((b_ ** 2).mean()))
    n_last = int(1.0 / env.dt)
    final_err = float(err[-n_last:].mean()) if finite else np.inf
    ok = bool(finite and final_err < 0.10 and err.max() < 0.5)
    m = dict(method=method, traj=traj_name, seed=seed, com=list(map(float, com)), mass=mass,
             rmse=float(np.sqrt((err ** 2).mean())) if finite else np.inf,
             max_err=float(err.max()), final_err=final_err,
             max_swing_deg=float(np.max(swing)) if swing else np.inf,
             max_tilt_deg=float(np.max(tilt)) if tilt else np.inf,
             slack_events=int(slack_events), ff_tension_rel_err=ff_rel, success=ok, finite=finite,
             t_step_mean_ms=float(np.mean(ts_time) * 1e3) if ts_time else 0.0,
             t_step_p95_ms=float(np.percentile(np.array(ts_time), 95) * 1e3) if ts_time else 0.0)
    if record:
        m["series"] = dict(P=P, R=R, tension=np.array(tr_T), ff_T=np.array(ff_T), tilt=np.array(tilt))
    return m
