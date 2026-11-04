"""P2d 測試：感測強度參數化（s=0/1 逐位等於 clean/moderate）、EKF-L、CON-ATT（一致性、局部性、掉包）。"""
import numpy as np
import torch

torch.set_num_threads(1)

from smpc.adapt.estimators import Geometry
from smpc.eval.sensing import PROFILES, get_profile, scaled_profile


def _same(a, b):
    ka = {k: v for k, v in a.items() if k not in ("profile", "series", "ident")}
    kb = {k: v for k, v in b.items() if k not in ("profile", "series", "ident")}
    assert ka == kb


def test_scaled_profile_rules():
    c, m, h = PROFILES["clean"], PROFILES["moderate"], PROFILES["harsh"]
    assert get_profile("s=0") == c and get_profile("s=1") == m and get_profile("s=1.0") == m
    assert get_profile("harsh") is h
    for s, d in ((0.25, 0), (0.5, 2), (1.0, 2), (1.74, 2), (1.75, 3), (2.5, 3)):
        assert scaled_profile(s).delay == d, s
    p = scaled_profile(1.75)
    assert np.isclose(p.sigma_a, 0.15 + 1.75 * 0.45) and np.isclose(p.bias_std, 0.175) and np.isclose(p.kerr, 0.0875)
    assert np.isclose(p.sigma_q, 0.005 + 1.75 * 0.005) and p.pos_std == 0.01
    p5 = scaled_profile(2.5)                      # 約等於 harsh
    assert abs(p5.bias_std - h.bias_std) / h.bias_std < 0.3 and abs(p5.kerr - h.kerr) / h.kerr < 0.3
    assert scaled_profile(0.5).pos_std == 0.005


def test_s0_s1_bitwise_equal_clean_moderate():
    from smpc.eval.episode_p2c import run_episode_p2c
    for name, alias in (("clean", "s=0"), ("moderate", "s=1")):
        a = run_episode_p2c("DOB", "circle", 2001, name, 1.0, {"fc": 4.0}, T=1.5, record=True)
        b = run_episode_p2c("DOB", "circle", 2001, alias, 1.0, {"fc": 4.0}, T=1.5, record=True)
        assert a["rmse"] == b["rmse"]
        assert np.array_equal(a["series"]["y_used"], b["series"]["y_used"])
        _same(a, b)
    a = run_episode_p2c("LOC-M", "circle", 2001, "moderate", 1.0, None, eta_comp="est", T=1.0)
    b = run_episode_p2c("LOC-M", "circle", 2001, "s=1", 1.0, None, eta_comp="est", T=1.0)
    _same(a, b)


def test_ekfl_recovers_bias_zeta_dm_synthetic():
    """合成資料：有激勵（旋轉的 a_ref 與 f）時 EKF-L 能同時估出 δm、ζ、b；不使用任何真值輸入。"""
    from smpc.adapt.ekf_local import EKFL
    from smpc.adapt.estimators import tension_model
    geom = Geometry(np.array([[0.3, 0, 0], [-0.15, 0.26, 0], [-0.15, -0.26, 0]]),
                    np.array([[1.0, 0, 0.8], [-0.5, 0.87, 0.8], [-0.5, -0.87, 0.8]]))
    rng = np.random.default_rng(0)
    m_q, sf = 1.0, 0.05
    dm, zeta, b = 0.2, np.array([0.04, -0.03, 0.02]), rng.normal(0, 0.15, (3, 3))
    ekf = EKFL(geom, m_q, sf, dict(q_b=1e-4))
    q = np.array([[0.2, 0, 1.0], [-0.1, 0.17, 1.0], [-0.1, -0.17, 1.0]]); q /= np.linalg.norm(q, axis=1, keepdims=True)
    for k in range(6000):
        t = k * 0.01
        a_ref = 3.0 * np.array([np.cos(t), np.sin(t), 0.0])
        f = np.array([[2.0 * np.cos(0.7 * t + i) , 2.0 * np.sin(0.7 * t + i), 3.5] for i in range(3)])
        y = np.zeros((3, 3))
        for i in range(3):
            T, _ = tension_model(geom, np.array([dm, 0, 0]), a_ref, i)
            y[i] = T * q[i] + zeta[i] * f[i] + m_q * b[i] + sf * rng.standard_normal(3)
        ekf.update(a_ref, q, f, y, np.ones(3, bool))
    assert np.allclose(ekf.zeta, zeta, atol=0.02)
    assert np.allclose(ekf.theta[:, 0], dm, atol=0.1)
    assert np.allclose(ekf.bias, b, atol=0.1)


def test_ekfl_skips_update_when_inactive():
    from smpc.adapt.ekf_local import EKFL
    geom = Geometry(np.eye(3) * 0.3, np.array([[1.0, 0, 0.8], [-0.5, 0.87, 0.8], [-0.5, -0.87, 0.8]]))
    ekf = EKFL(geom, 1.0, 0.1)
    q = np.tile([0, 0, 1.0], (3, 1))
    ekf.update(np.zeros(3), q, np.ones((3, 3)), np.ones((3, 3)), np.array([False, True, True]))
    assert np.all(ekf.x[0] == 0) and np.any(ekf.x[1] != 0)


def _record_cen(T=4.0, profile="moderate", seed=2001):
    from smpc.eval.episode_p2c import run_episode_p2c
    r = run_episode_p2c("CEN-ATT", "circle", seed, profile, 1.0, None, eta_comp="est", T=T, ident=True)
    return r["ident"]


def _geom_L():
    from smpc.eval.episode import make_env
    env = make_env(3, 1.0, (0.0, 0.0, 0.0))
    return Geometry.from_env(env, 30.0, 1.0), env.L


def test_con_att_matches_cen_att_ideal_many_iterations():
    """理想通訊（p_drop=0）、迭代次數很大 -> 每機 θ̂ 和 ζ̂ 收斂到 CEN-ATT（誤差 < 1%）。同一串量測資料離線重放。"""
    from smpc.adapt.attitude import ReplayCEN
    from smpc.adapt.con_att import ConATT
    ID = _record_cen()
    geom, L = _geom_L()
    sf, sq = 0.6 * 1.0, 0.01
    cen = ReplayCEN(geom, L, "att", lam=0.9995, sigma_f=sf, sigma_q=sq)
    con = ConATT(geom, L, lam=0.9995, sigma_f=sf, sigma_q=sq, iters=60)
    worst = 0.0
    for k in range(len(ID["a_ref"])):
        args = (ID["a_ref"][k], ID["q"][k], ID["f"][k], ID["y"][k], ID["active"][k].astype(bool), ID["pq"][k])
        cen.update(*args)
        con.update(*args)
        if k >= 50 and k % 25 == 0:
            ref = np.concatenate([cen.theta, cen.zeta])
            for i in range(3):
                worst = max(worst, np.linalg.norm(con.x[i] - cen.x) / np.linalg.norm(ref))
    assert con.pose_ok_frac[-1] == 1.0
    assert worst < 0.01, worst
    # 預設 2 次迭代：不要求一致，但必須有限且仍在同一量級（診斷用上界）
    con2 = ConATT(geom, L, lam=0.9995, sigma_f=sf, sigma_q=sq, iters=2)
    for k in range(len(ID["a_ref"])):
        con2.update(ID["a_ref"][k], ID["q"][k], ID["f"][k], ID["y"][k], ID["active"][k].astype(bool), ID["pq"][k])
    assert np.isfinite(con2.x).all()


def test_con_att_uses_only_neighbor_relayed_data():
    """局部性：關掉量測轉傳（relay_rounds=0）、共識 1 次時，同一步內機 1 的估計不受機 3（2 跳外）的原始量測影響；
    打開轉傳 2 輪後才會受影響。"""
    from smpc.adapt.con_att import ConATT
    ID = _record_cen(T=1.0)
    geom, L = _geom_L()

    def run(rounds, perturb):
        con = ConATT(geom, L, sigma_f=0.6, sigma_q=0.01, iters=1, relay_rounds=rounds)
        for k in range(1):                       # 單步：多步時資訊會經共識一跳一跳擴散（合法），這裡只驗原始量測不跨兩跳
            y = ID["y"][k].copy(); pq = ID["pq"][k].copy()
            if perturb:
                y[2] += 5.0; pq[2] += 0.2
            con.update(ID["a_ref"][k], ID["q"][k], ID["f"][k], y, ID["active"][k].astype(bool), pq)
        return con.x[0].copy()
    # 機 1 的本地資料不變；機 3 的資料經 0 輪轉傳 + 1 次共識到不了機 1
    assert np.array_equal(run(0, False), run(0, True))
    assert not np.allclose(run(2, False), run(2, True))


def test_con_att_dropout_behaviour():
    """p_drop：同 seed 可重現；p=1 時各機互不通訊（機 1 估計與機 3 資料無關）；p=0.5 有限。"""
    from smpc.adapt.con_att import ConATT
    ID = _record_cen(T=1.0)
    geom, L = _geom_L()

    def run(p, perturb, seed=7):
        con = ConATT(geom, L, sigma_f=0.6, sigma_q=0.01, p_drop=p, seed=seed)
        for k in range(len(ID["a_ref"])):
            y = ID["y"][k].copy()
            if perturb:
                y[2] += 5.0
            con.update(ID["a_ref"][k], ID["q"][k], ID["f"][k], y, ID["active"][k].astype(bool), ID["pq"][k])
        return con
    a, b = run(0.5, False), run(0.5, False)
    assert np.array_equal(a.x, b.x) and np.isfinite(a.x).all()
    assert not np.array_equal(a.x, run(0.5, False, seed=8).x)
    assert np.array_equal(run(1.0, False).x[0], run(1.0, True).x[0])       # 完全掉包：機 1 看不到機 3
    assert not np.array_equal(run(0.0, False).x[0], run(0.0, True).x[0])


def test_new_methods_run_and_dispatch():
    from smpc.eval.episode_p2c import run_episode_p2c
    for m, hp in (("EKF-L", {}), ("CON-ATT", {"p_drop": 0.5})):
        r = run_episode_p2c(m, "circle", 2001, "s=1.75", 1.0, hp, eta_comp="est", T=1.0)
        assert r["finite"] and np.isfinite(r["rmse"]), m


# ---------------------------------------------------------------- PREREG 前修正：relay_rounds、ζ̂ 記錄、EKF-L+GEO
def _geom_L_n(n):
    from smpc.eval.episode import make_env
    env = make_env(n, 1.0, (0.0, 0.0, 0.0))
    return Geometry.from_env(env, 30.0, 1.0), env.L


def _rand_packets(n, rng):
    return dict(a_ref=rng.normal(0, 1, 3), q=rng.normal(0, 0.1, (n, 3)) + [0, 0, -1.0], f=rng.normal(0, 1, (n, 3)),
                y=rng.normal(0, 1, (n, 3)), active=np.ones(n, bool), pq=rng.normal(0, 1, (n, 3)))


def test_con_att_relay_rounds_default_is_graph_diameter():
    from smpc.adapt.con_att import ConATT
    for n in (3, 4, 6):
        geom, L = _geom_L_n(n)
        assert ConATT(geom, L).relay_rounds == n - 1
        assert ConATT(geom, L, relay_rounds=1).relay_rounds == 1
        assert ConATT(geom, L, relay_rounds=0).relay_rounds == 0


def test_every_agent_gets_full_team_packets_after_relay_n4_n6():
    from smpc.adapt.con_att import ConATT
    from smpc.adapt.ekf_geo import EKFLGeo
    rng = np.random.default_rng(0)
    for n in (3, 4, 6):
        geom, L = _geom_L_n(n)
        con, geo = ConATT(geom, L), EKFLGeo(geom, L, 1.0, 0.1)
        for _ in range(3):
            P = _rand_packets(n, rng)
            con.update(P["a_ref"], P["q"], P["f"], P["y"], P["active"], P["pq"])
            geo.update(P["a_ref"], P["q"], P["f"], P["y"], P["active"], P["pq"])
            for est, w in ((con, 13), (geo, 6)):
                assert np.all(est.stamp == est.k)                           # 每機手上每個來源的封包都是當步的
                for i in range(n):
                    for o in range(n):                                       # 內容等於來源自己的封包
                        assert np.array_equal(est.data[i, o], est.data[o, o])
                assert est.data.shape[-1] == w
            assert np.array_equal(con.data[0, n - 1, 0:3], P["pq"][n - 1])
    # 對照：N=4 若只轉傳 2 輪（舊的固定值），端點機拿不到對面的封包
    geom, L = _geom_L_n(4)
    old = ConATT(geom, L, relay_rounds=2)
    P = _rand_packets(4, rng)
    old.update(P["a_ref"], P["q"], P["f"], P["y"], P["active"], P["pq"])
    assert old.stamp[0, 3] < old.k and old.stamp[3, 0] < old.k


def test_con_att_n3_unchanged_bitwise_vs_stored_sweep():
    """relay_rounds 修正後 N=3 逐位不變：sweep.jsonl 裡 CON-ATT s=1 circle seed 2200 的 rmse（修正前跑的）。"""
    from smpc.eval.episode_p2c import run_episode_p2c
    r = run_episode_p2c("CON-ATT", "circle", 2200, "s=1", 1.0, {"lam": 0.9998, "prior": 2.0}, eta_comp="est")
    assert r["rmse"] == 0.031139830673288314
    assert "zeta_rms_h2" in r and np.isfinite(r["zeta_rms_h2"])


def test_zeta_logging_is_record_only_bitwise_ekfl_locm():
    """ζ̂ 誤差只記錄：EKF-L、LOC-M 的 rmse 與加入記錄前跑出的 sweep.jsonl 值逐位相同；DOB（none 路徑）沒有 ζ̂ 欄位。"""
    from smpc.eval.episode_p2c import run_episode_p2c
    r = run_episode_p2c("EKF-L", "circle", 2201, "s=1", 1.0, {"q_b": 0.0005, "r_scale": 10.0}, eta_comp="est")
    assert r["rmse"] == 0.03180069321015727 and r["zeta_rms_h2"] > 0
    r = run_episode_p2c("LOC-M", "circle", 2201, "s=1", 1.0, {"lam": 0.998, "prior": 1.0}, eta_comp="est")
    assert r["rmse"] == 0.030926502351704598
    r = run_episode_p2c("DOB", "circle", 2201, "s=1", 1.0, {"fc": 16.0})
    assert r["rmse"] == 0.03196378489987269 and "zeta_rms_h2" not in r


def test_ekf_geo_uses_only_teammate_positions():
    """資訊條件：機 i 的估計只依賴自己的 y_i、f_i、active_i 與全隊的 (pq, q)。擾動隊友的 y_j、f_j、active_j 不改變 x_i；擾動隊友位置會改。"""
    from smpc.adapt.ekf_geo import EKFLGeo
    geom, L = _geom_L_n(3)
    rng = np.random.default_rng(1)
    seq = [_rand_packets(3, rng) for _ in range(30)]

    def run(mut):
        e = EKFLGeo(geom, L, 1.0, 0.1)
        for P in seq:
            P = {k: np.array(v, copy=True) for k, v in P.items()}
            mut(P)
            e.update(P["a_ref"], P["q"], P["f"], P["y"], P["active"], P["pq"])
        return e.x.copy()

    base = run(lambda P: None)

    def other_meas(P):
        P["y"][1:] += 3.0; P["f"][1:] *= 1.7; P["active"][1:] = False
    assert np.array_equal(run(other_meas)[0], base[0])             # 隊友的量測（y、f、active）完全不影響機 0
    assert not np.array_equal(run(other_meas)[1], base[1])         # 自己的量測影響自己（檢查擾動有效）

    def other_pos(P):
        P["pq"][2] += 0.3
    assert not np.array_equal(run(other_pos)[0], base[0])          # 隊友位置影響機 0（幾何通道有接上）
    # 估計器不吃 payload 真值：介面只有 (a_ref, q, f, y, active, pq)
    import inspect
    assert list(inspect.signature(EKFLGeo.update).parameters) == ["self", "a_ref", "q", "f", "y", "active", "pq"]


def test_ekf_geo_never_reads_teammate_estimates():
    """各機的 EKF 狀態/協方差互相獨立：把機 1、2 的內部狀態換掉，機 0 的下一步結果不變。"""
    from smpc.adapt.ekf_geo import EKFLGeo
    geom, L = _geom_L_n(3)
    rng = np.random.default_rng(2)
    seq = [_rand_packets(3, rng) for _ in range(10)]
    a, b = EKFLGeo(geom, L, 1.0, 0.1), EKFLGeo(geom, L, 1.0, 0.1)
    for P in seq[:5]:
        a.update(P["a_ref"], P["q"], P["f"], P["y"], P["active"], P["pq"])
        b.update(P["a_ref"], P["q"], P["f"], P["y"], P["active"], P["pq"])
    b.x[1:] += 5.0; b.P[1:] *= 7.0
    for P in seq[5:]:
        a.update(P["a_ref"], P["q"], P["f"], P["y"], P["active"], P["pq"])
        b.update(P["a_ref"], P["q"], P["f"], P["y"], P["active"], P["pq"])
    assert np.array_equal(a.x[0], b.x[0]) and np.array_equal(a.P[0], b.P[0])


def test_ekf_geo_episode_runs_and_logs_zeta():
    from smpc.eval.episode_p2c import run_episode_p2c
    r = run_episode_p2c("EKF-L+GEO", "circle", 2001, "s=1", 1.0, {"q_b": 5e-4, "r_scale": 10.0}, eta_comp="est", T=1.5)
    assert r["finite"] and np.isfinite(r["rmse"]) and np.isfinite(r["zeta_rms_h2"])
