"""P2c 閉迴路 episode：P2b 的執行器 + 感測 profile（sensing.py）+ 混合法 HYB / HYB-local + ORACLE 診斷。

與 episode_p2b 的差別只在量測層與新方法；控制器、估計器、指標定義不變。clean profile 且 eta_comp 同 P2b 時，結果與 P2b 相同
（sens 流的抽樣順序不變，見 tests/test_p2c.py 與 results/p2c/clean_repro.md）。
method: NOM | ORACLE | DOB | LS | LSN | HYB-local | CEN-ATT | HYB | LOC-M（本地 LocalDm 模型前饋＋可選本地 ζ̂ 補償，無殘差通道）
        | EKF-L（P2d 本地：每機 EKF，狀態 [δm, ζ_i, IMU 偏差 b_i]，hp 見 adapt/ekf_local.py）
        | EKF-L+GEO（拆解對照：EKF-L + 經轉傳取得的隊友位置/繩向，無任何隊友估計或量測；見 adapt/ekf_geo.py）
        | CON-ATT（P2d 分散式：CEN-ATT 在 path 圖上的 HCMCI，hp: p_drop、iters；見 adapt/con_att.py）
profile: clean | moderate | harsh | "s=<比例>"（P2d 感測強度掃描，見 sensing.scaled_profile）
hp: 方法超參數 dict。DOB/LS/LSN：fc。HYB/HYB-local：fc（修正量低通）、tau（修正量衰減時間常數）。
oracle_acc: 診斷（只對 ORACLE）：前饋張力用的加速度 = a_ref + (payload 實際加速度 - a_ref)（pilot 診斷用）。
"""
from __future__ import annotations

import numpy as np

from smpc.adapt.estimators import Geometry, tension_model
from smpc.adapt.hybrid import HybridTension
from smpc.adapt.local_baselines import DOBHold, LoadShare, LocalDm
from smpc.control.pd import quat_to_rot
from smpc.control.suspended_mppi import MPPIConfig, SuspendedMPPITeam
from smpc.control.trajectories import Traj
from smpc.eval.episode import DURATION, M_NOM, make_env, rope_dirs
from smpc.eval.episode_est import SLACK_DEPTH, SLACK_TMIN, TH_CLIP, _unit, draw_truth
from smpc.eval.sensing import SensorModel, get_profile

METHODS = ("NOM", "ORACLE", "DOB", "LS", "LSN", "HYB-local", "CEN-ATT", "HYB", "LOC-M", "EKF-L", "CON-ATT", "EKF-L+GEO")
LEAD = 10


def run_episode_p2c(method, traj_name, seed, profile="clean", speed=1.0, hp=None, eta_comp="none", n=3,
                    cfg: MPPIConfig | None = None, T=None, record=False, lam=0.9995, oracle_acc=False, ident=False):
    assert method in METHODS
    hp = hp or {}
    lam = hp.get("lam", lam)              # P2d 調參：遺忘因子、先驗縮放（預設不變）
    psc = hp.get("prior", 1.0)
    prof = get_profile(profile)
    th_true, _, _ = draw_truth(seed, False, n)
    init_rng = np.random.default_rng([seed, 2])
    sens = np.random.default_rng([seed, 3])
    sm = SensorModel(prof, seed, n, 0.01)
    cfg = cfg or MPPIConfig.load("configs/mppi_frozen.json")
    mass, com = M_NOM + th_true[0], (th_true[1], th_true[2], 0.0)
    env = make_env(n, mass, com)
    traj = Traj(traj_name, speed=speed)
    T = T or DURATION[traj_name]
    p0 = traj.ref(0.0)[0][0] + init_rng.normal(0, [0.03, 0.03, 0.02])
    o = env.reset(payload_pos=p0)
    env.eta = np.tile(sm.k_true[:, None], (1, 4))          # 真實推力係數（估計器與 DOB 不知道）
    env.kick_payload(init_rng.normal(0, 0.1, 3))
    th0 = th_true if method == "ORACLE" else np.zeros(3)
    team = SuspendedMPPITeam(env, traj, cfg, theta_hat=th0, m_nom=M_NOM, seed=seed)
    geom = Geometry.from_env(env, cfg.cone_deg, M_NOM, cfg.w_moment)
    dt, m_q, g = env.dt, env.p.mass, env.p.g
    sa, sq = prof.sigma_a, prof.sigma_q
    sigma_f = sa * m_q
    ez = np.array([0, 0, g])
    est = loc = dob = ls = hyb = None
    if method == "CEN-ATT" or method == "HYB":
        from smpc.adapt.attitude import ReplayCEN
        est = ReplayCEN(geom, env.L, "att", lam=lam, sigma_f=sigma_f, sigma_q=sq, prior_scale=psc)
    if method == "CON-ATT":
        from smpc.adapt.con_att import ConATT
        est = ConATT(geom, env.L, lam=lam, sigma_f=sigma_f, sigma_q=sq, iters=hp.get("iters", 2),
                     p_drop=hp.get("p_drop", 0.0), seed=seed, prior_scale=psc,
                     relay_rounds=hp.get("relay_rounds"))
    if method == "EKF-L":
        from smpc.adapt.ekf_local import EKFL
        loc = EKFL(geom, m_q, sigma_f, hp)
    if method == "EKF-L+GEO":
        from smpc.adapt.ekf_geo import EKFLGeo
        loc = EKFLGeo(geom, env.L, m_q, sigma_f, hp, relay_rounds=hp.get("relay_rounds"))
    if method in ("HYB-local", "LOC-M"):   # LOC-M：同一個 LocalDm，但不接 hyb（無 DOB 殘差通道），走 team.set_theta
        loc = LocalDm(geom, prior_dm=0.5, lam=lam, sigma_f=sigma_f, prior_scale=psc)
    if method == "DOB":
        dob = DOBHold(n, hp["fc"], dt)
        for i in range(n):
            team.set_tension_fn(i, dob.force_fn(i))
    elif method in ("LS", "LSN"):
        ls = LoadShare(n, hp["fc"], dt, M_NOM, g, geom, nominal=(method == "LSN"))
        for i in range(n):
            team.set_tension_fn(i, ls.force_fn(i))
    elif method in ("HYB", "HYB-local"):
        hyb = HybridTension(geom, n, hp["fc"], hp["tau"], dt)
        for i in range(n):
            team.set_tension_fn(i, hyb.force_fn(i))
    if eta_comp == "oracle":
        for i in range(n):
            team.set_eta(i, sm.k_true[i])

    def meas_dir():
        return _unit(rope_dirs(env) + sq * sens.standard_normal((n, 3)))

    q_next = meas_dir()
    steps = int(round(T / dt))
    P, R, tilt, swing, tr_T, pred10, pred0 = [], [], [], [], [], [], []
    clipn, inactive, finite, prev, slack_events = 0, 0, True, False, 0
    a_pay_f, v_pay_prev = None, np.array(o["payload_state"][7:10])
    rec = {k: [] for k in ("y_used", "y_now", "tm")} if record else None
    th_hist, z_hist = [], []
    idr = {k: [] for k in ("zeta", "theta", "a_ref", "q", "f", "y", "active", "pq", "tm", "Lam")} if ident else None
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
        a_imu = a_spec + sm.step_bias() + sa * sens.standard_normal((n, 3))
        b3 = _unit(np.array([quat_to_rot(qs0[i, 3:7])[:, 2] + quat_to_rot(qs1[i, 3:7])[:, 2] for i in range(n)]))
        f = Uc.sum(1)[:, None] * b3                       # 以為的推力 = 指令（名義係數 1）
        q_est = _unit(q_now + q_next)
        y = m_q * a_imu - f
        tm = t + 0.5 * dt
        a_ref = traj.ref(tm)[2][0]
        pq_mid = sm.pos_meas(0.5 * (qs0[:, :3] + qs1[:, :3]))
        pa_ref = traj.ref(tm)[0][0][None] + env.attach_body
        T_pred = np.array([tension_model(geom, np.zeros(3), a_ref, i)[0] for i in range(n)])
        active = (np.linalg.norm(pq_mid - pa_ref, axis=1) >= env.L - SLACK_DEPTH) & (T_pred >= SLACK_TMIN)
        inactive += int((~active).sum())
        bundle = dict(y=y, f=f, q=q_est, a_ref=a_ref, active=active, pq=pq_mid, tm=tm)
        if hyb is not None and eta_comp != "none":
            # HYB 殘差通道：控制器已把指令除以 η̂，它「以為送出」的推力是 η̂·Uc（不是名義 1·Uc）。
            # 殘差必須相對這個推力量，否則 (k-1)·f 這項被 η̂ 補償一次、又被 DOB 殘差補償一次（雙重修正）。
            # ζ̂ 估計器仍用名義 f（y = Tq + ζ f 的量測模型），所以只換 HYB 用的 y。
            eh = np.array([a.eta_hat for a in team.agents])[:, None]
            bundle["y_hyb"] = y + (1.0 - eh) * f
        bd = sm.delay_bundle(bundle)
        if record:
            rec["y_used"].append(bd["y"]); rec["y_now"].append(y); rec["tm"].append(bd["tm"])
        theta_now = zeta_now = None
        if est is not None:
            est.update(bd["a_ref"], bd["q"], bd["f"], bd["y"], bd["active"], bd["pq"])
            theta_now, zeta_now = (est.theta, est.zeta) if method == "CON-ATT" else (np.tile(est.theta, (n, 1)), est.zeta)
        elif loc is not None:
            if method == "EKF-L+GEO":
                loc.update(bd["a_ref"], bd["q"], bd["f"], bd["y"], bd["active"], bd["pq"])
            else:
                loc.update(bd["a_ref"], bd["q"], bd["f"], bd["y"], bd["active"])
            theta_now, zeta_now = loc.theta, loc.zeta
        if dob is not None:
            dob.update(bd["y"], bd["active"])
        if ls is not None:
            ls.update(bd["a_ref"], bd["q"], bd["y"], bd["active"])
        if theta_now is not None:
            for i in range(n):
                th_c = np.clip(theta_now[i], *TH_CLIP)
                clipn += int(np.any(th_c != theta_now[i]))
                if hyb is not None:
                    hyb.set_theta(i, th_c)
                else:
                    team.set_theta(i, th_c)
            th_hist.append(theta_now.copy())
            z_hist.append(np.array(zeta_now, float).copy())   # 只記錄 ζ̂，不回饋控制
        if ident and theta_now is not None:   # 只記錄，不回饋（identification 驗證用）
            for k_, v_ in (("zeta", zeta_now), ("theta", theta_now), ("a_ref", bd["a_ref"]), ("q", bd["q"]), ("f", bd["f"]),
                           ("y", bd["y"]), ("active", bd["active"]), ("pq", bd["pq"]), ("tm", bd["tm"])):
                idr[k_].append(np.array(v_, float).copy())
            if k % 50 == 49:
                idr["Lam"].append((est or loc).rls.Lambda.copy())
        if hyb is not None:
            hyb.update(bd.get("y_hyb", bd["y"]), bd["a_ref"], bd["q"])
        if eta_comp == "est" and zeta_now is not None:
            for i in range(n):
                team.set_eta(i, 1.0 + zeta_now[i])
        if oracle_acc:   # 診斷：ORACLE 用 payload 實際加速度（真值，濾波）修正前饋張力的加速度
            v_now = np.array(ps[7:10])
            a_raw = (v_now - v_pay_prev) / dt
            v_pay_prev = v_now
            a_pay_f = a_raw if a_pay_f is None else a_pay_f + 0.3 * (a_raw - a_pay_f)
            for a in team.agents:
                a.a_bias = a_pay_f - a_ref
        P.append(ps[:3]); R.append(traj.ref(t + dt)[0][0])
        w_, x_, y_, z_ = ps[3:7]
        tilt.append(np.degrees(np.arccos(np.clip(1 - 2 * (x_ * x_ + y_ * y_), -1, 1))))
        cen = qs1[:, :3].mean(0) - ps[:3]
        swing.append(np.degrees(np.arctan2(np.linalg.norm(cen[:2]), cen[2])))
        sl = bool((o["tension_truth"] < 0.02).any())
        slack_events += int(sl and not prev)
        prev = sl
        tr_T.append(o["tension_truth"].copy())
    P, R = np.array(P), np.array(R)
    err = np.linalg.norm(P - R, axis=1) if len(P) else np.array([np.inf])
    n_last = int(1.0 / dt)
    final_err = float(err[-n_last:].mean()) if finite else np.inf
    ok = bool(finite and final_err < 0.10 and err.max() < 0.5)
    m = dict(method=method, profile=profile, traj=traj_name, speed=speed, seed=seed, eta_comp=eta_comp, hp=hp,
             theta_true=th_true.tolist(), k_true=sm.k_true.tolist(),
             rmse=float(np.sqrt((err ** 2).mean())) if finite else np.inf, max_err=float(err.max()), final_err=final_err,
             max_tilt_deg=float(np.max(tilt)) if tilt else np.inf, max_swing_deg=float(np.max(swing)) if swing else np.inf,
             slack_events=int(slack_events), success=ok, finite=finite, clip_frac=clipn / max(1, steps * n),
             inactive_frac=inactive / max(1, steps * n))
    if finite and len(tr_T) > 300:
        TT = np.array(tr_T)
        p10, p0_ = np.array(pred10), np.array(pred0)
        lead = 50
        ks = np.arange(200, len(TT) - lead)
        m["tpe10"] = float(np.sqrt(((p10[ks] - TT[ks + lead - 1]) ** 2).mean()))
        m["tpe0"] = float(np.sqrt(((p0_[ks] - TT[ks]) ** 2).mean()))
        m["tension_rms"] = float(np.sqrt((TT[ks] ** 2).mean()))
    if th_hist:
        e = np.array(th_hist) - th_true
        ix = min(499, len(e) - 1)
        m["dm_err@5"] = float(np.abs(e[ix, :, 0]).mean())
        m["dc_err@5"] = float(np.linalg.norm(e[ix, :, 1:], axis=1).mean())
    if z_hist:   # ζ̂ 誤差（只記錄）：真值 ζ_i = k_true_i - 1；後半段（含）所有機的 RMS，與全程 RMS
        ze = np.array(z_hist) - (sm.k_true - 1.0)
        m["zeta_rms_h2"] = float(np.sqrt((ze[len(ze) // 2:] ** 2).mean()))
        m["zeta_rms_all"] = float(np.sqrt((ze ** 2).mean()))
    if ident:
        m["ident"] = {k_: np.array(v) for k_, v in idr.items()}
    if record:
        m["series"] = {k_: np.array(v) for k_, v in rec.items()}
    return m
