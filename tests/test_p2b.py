"""P2b 測試：DOB 靜態懸停、LOADSHARE 預測、η 補償方向。"""
import numpy as np
import torch

torch.set_num_threads(1)

from smpc.adapt.estimators import Geometry, tension_model
from smpc.adapt.local_baselines import DOBHold, LoadShare, LocalDm
from smpc.control.trajectories import Traj
from smpc.eval.episode import M_NOM, PDStatic, make_env, rope_dirs


def test_dob_static_hover_near_nominal_tension():
    """PD 靜態前饋懸停：DOB 估的繩力沿繩分量 ≈ 名義張力（θ=0、a_ref=0 的 tension_model）。無雜訊。"""
    env = make_env(3, M_NOM, (0, 0, 0))
    traj = Traj("hover")
    o = env.reset(payload_pos=traj.ref(0.0)[0][0])
    ctrl = PDStatic(env, traj, M_NOM, (0, 0, 0))
    geom = Geometry.from_env(env, 30.0, M_NOM, 1.0)
    dob = DOBHold(3, fc=2.0, dt=env.dt)
    ez = np.array([0, 0, env.p.g])
    from smpc.control.pd import quat_to_rot
    for k in range(400):
        qs0 = o["quad_states"]
        U = ctrl.act(k * env.dt, qs0)
        o = env.step(U)
        qs1 = o["quad_states"]
        b3 = np.array([quat_to_rot(qs0[i, 3:7])[:, 2] + quat_to_rot(qs1[i, 3:7])[:, 2] for i in range(3)])
        b3 /= np.linalg.norm(b3, axis=1, keepdims=True)
        f = U.sum(1)[:, None] * b3
        y = env.p.mass * ((qs1[:, 7:10] - qs0[:, 7:10]) / env.dt + ez) - f
        dob.update(y)
    q = rope_dirs(env)
    T_dob = np.einsum("ij,ij->i", dob.F, q)
    T_nom = np.array([tension_model(geom, np.zeros(3), np.zeros(3), i)[0] for i in range(3)])
    assert np.all(np.abs(T_dob - T_nom) < 0.15 * T_nom + 0.1), (T_dob, T_nom)


def _synthetic(traj_name, theta):
    env = make_env(3)
    geom = Geometry.from_env(env, 30.0, M_NOM, 1.0)
    tr = Traj(traj_name)
    q = -(geom.offsets - geom.attach_body)
    q = -q / np.linalg.norm(q, axis=1, keepdims=True)        # 機 -> 掛點
    return geom, tr, q


def _ls_err(nominal, a_fn, th, fc=2.0):
    geom, tr, q = _synthetic("circle", th)
    ls = LoadShare(3, fc=fc, dt=0.01, m_nom=M_NOM, g=9.81, geom=geom, nominal=nominal)
    errs, Ts = [], []
    for k in range(1000):
        t = k * 0.01
        a = a_fn(t)[0]
        y = np.array([tension_model(geom, th, a, i)[0] * q[i] for i in range(3)])
        ls.update(a, q, y)
        if 300 <= k < 900:
            ts = np.array([t + 0.5])
            a_f = a_fn(ts)
            for i in range(3):
                Tp = ls.force_fn(i)(t, ts, a_f, q[i])[0] @ q[i]
                Tt = tension_model(geom, th, a_f[0], i)[0]
                errs.append(Tp - Tt); Ts.append(Tt)
    return np.sqrt(np.mean(np.square(errs))) / np.mean(Ts)


def test_loadshare_prediction_error_small_with_true_theta_no_noise():
    """θ=(δm,0,0) 真值、無雜訊、y = T_model(θ) q。
    - LOADSHARE-N（名義張力映射 x 分配比）：圓軌跡上預測 0.5 s 後張力的誤差應 ~0。
    - LOADSHARE（|a+g| 版）：加速度只在垂直方向時分配比不變 -> 誤差 ~0（驗證實作）；圓軌跡上分配比隨加速度方向變，
      有結構性誤差（實測約 19%），這是規格公式本身的限制，只檢查它不爆掉（< 30%）。"""
    th = np.array([0.3, 0.0, 0.0])
    tr = Traj("circle")
    circ = lambda t: tr.ref(t)[2]
    vert = lambda t: np.stack([np.zeros_like(np.atleast_1d(t)), np.zeros_like(np.atleast_1d(t)), 1.5 * np.sin(1.0 * np.atleast_1d(t))], 1)
    assert _ls_err(True, circ, th) < 0.02
    assert _ls_err(False, vert, th) < 0.03
    assert _ls_err(False, circ, th) < 0.30


def test_local_dm_recovers_dm_noise_free():
    geom, tr, q = _synthetic("circle", None)
    th = np.array([0.3, 0.0, 0.0])
    est = LocalDm(geom, prior_dm=0.5, sigma_f=0.15)
    for k in range(1500):
        a = tr.ref(k * 0.01)[2][0]
        y = np.array([tension_model(geom, th, a, i)[0] * q[i] for i in range(3)])
        est.update(a, q, np.zeros((3, 3)) + 1e-9, y)
    assert np.all(np.abs(est.theta[:, 0] - 0.3) < 0.03), est.theta[:, 0]


def test_eta_compensation_direction():
    """真實 η=0.75（一台）：η 補償開（oracle η）的 RMSE 明顯小於關。短 episode（3 s）。"""
    from smpc.eval.episode_p2b import run_episode_p2b
    off = run_episode_p2b("ORACLE", "circle", 2000, eta_fault=True, eta_comp="none", T=3.0)
    on = run_episode_p2b("ORACLE", "circle", 2000, eta_fault=True, eta_comp="oracle", T=3.0)
    assert on["rmse"] < 0.5 * off["rmse"], (on["rmse"], off["rmse"])
    # 補償開時送出的指令應是 1/η̂ 倍
    assert on["eta_hat_final"][on["k_fault"]] == 0.75
