"""P2c 測試：clean profile 量測與 P2b 一致、延遲確實落後、HYB 在 r=0 時等於模型預測、感測統計量。"""
import numpy as np
import torch

torch.set_num_threads(1)

from smpc.adapt.estimators import Geometry
from smpc.adapt.hybrid import HybridTension
from smpc.control.suspended_mppi import MPPIConfig, SuspendedMPPITeam
from smpc.control.trajectories import Traj
from smpc.eval.episode import M_NOM, make_env
from smpc.eval.sensing import PROFILES, SensorModel


def test_clean_profile_measurement_matches_p2b():
    from smpc.eval.episode_p2b import run_episode_p2b
    from smpc.eval.episode_p2c import run_episode_p2c
    a = run_episode_p2b("DOB", "circle", 2000, fc=2.0, T=2.0, record=True)
    b = run_episode_p2c("DOB", "circle", 2000, "clean", 1.0, {"fc": 2.0}, T=2.0, record=True)
    assert np.allclose(a["series"]["y"], b["series"]["y_now"], atol=1e-12)
    assert np.allclose(b["series"]["y_used"], b["series"]["y_now"])      # clean 沒有延遲
    assert a["rmse"] == b["rmse"]


def test_delay_profile_measurement_lags():
    from smpc.eval.episode_p2c import run_episode_p2c
    r = run_episode_p2c("DOB", "circle", 2000, "moderate", 1.0, {"fc": 4.0}, T=1.5, record=True)
    yu, yn = r["series"]["y_used"], r["series"]["y_now"]
    d = PROFILES["moderate"].delay
    assert d == 2
    assert np.allclose(yu[d:], yn[:-d])                  # 估計器看到的 = d 步前的量測
    assert not np.allclose(yu[d:], yn[d:])
    assert np.allclose(r["series"]["tm"][d:], np.arange(len(yn) - d) * 0.01 + 0.005)   # 量測時間戳記落後 d 步


def test_delay_buffer_unit():
    sm = SensorModel(PROFILES["harsh"], 0, 3, 0.01)
    out = [sm.delay_bundle(dict(k=k))["k"] for k in range(8)]
    assert out == [0, 0, 0, 0, 1, 2, 3, 4]               # delay=3；開頭不足時用最早的一包


def test_sensor_statistics_and_thrust_coefficient():
    for name, kerr, bstd in (("moderate", 0.05, 0.1), ("harsh", 0.10, 0.2)):
        ks, bs = [], []
        for s in range(200):
            sm = SensorModel(PROFILES[name], s, 3, 0.01)
            ks.append(sm.k_true); bs.append(sm.bias.ravel())
            for _ in range(5):
                sm.step_bias()
        ks = np.array(ks).ravel(); bs = np.concatenate(bs)
        assert ks.min() >= 1 - kerr and ks.max() <= 1 + kerr and ks.std() > 0.5 * kerr / np.sqrt(3)
        assert abs(bs.std() - bstd) < 0.15 * bstd
    sm = SensorModel(PROFILES["clean"], 0, 3, 0.01)
    assert np.all(sm.k_true == 1) and np.all(sm.bias == 0)
    # 穩態：長時間後標準差仍約 bias_std
    sm = SensorModel(PROFILES["moderate"], 1, 200, 0.01)
    sm.bias[:] = 0
    for _ in range(6000):
        sm.step_bias()
    assert abs(sm.bias.std() - 0.1) < 0.02


def test_thrust_coefficient_acts_on_env_not_on_assumed_thrust():
    from smpc.eval.episode_p2c import run_episode_p2c
    # 懸停型短 episode：真實推力係數進環境；f（以為的推力）= 指令。用 moderate 的 k_true 檢查 meta 即可
    r = run_episode_p2c("ORACLE", "circle", 2001, "moderate", 1.0, T=0.5)
    k = np.array(r["k_true"])
    assert np.all(np.abs(k - 1) <= 0.05) and np.any(k != 1)


def test_hybrid_equals_model_prediction_when_r_zero():
    cfg = MPPIConfig.load("configs/mppi_frozen.json")
    env = make_env(3, M_NOM, (0, 0, 0))
    traj = Traj("circle", speed=1.5)
    team = SuspendedMPPITeam(env, traj, cfg, theta_hat=np.zeros(3), m_nom=M_NOM, seed=0)
    geom = Geometry.from_env(env, cfg.cone_deg, M_NOM, cfg.w_moment)
    hyb = HybridTension(geom, 3, fc=4.0, tau=0.5, dt=0.01)
    th = np.array([0.25, 0.01, -0.02])
    q = np.array([0.1, -0.2, 0.97]); q = q / np.linalg.norm(q)
    for i in range(3):
        team.set_theta(i, th)
        hyb.set_theta(i, th)
    t = 1.3
    ts = t + cfg.dt_plan * np.arange(cfg.H + 1)
    a = traj.ref(ts)[2]
    for i in range(3):
        _, _, _, _, _, T_native = team.agents[i]._feedforward(t, q)
        F = hyb.force_fn(i)(t, ts, a, q)                  # r = 0（初始）
        assert np.allclose(F, T_native[:, None] * q[None], atol=1e-9)
    # r != 0：修正量以 exp(-s/tau) 衰減，s = 0 時整個加上
    hyb.r[:] = np.array([1.0, 2.0, 3.0])
    F0 = hyb.force_fn(0)(t, ts, a, q)
    base = hyb.model_T(0, a)[:, None] * q[None]
    assert np.allclose(F0[0] - base[0], [1, 2, 3])
    assert np.allclose(F0[10] - base[10], np.array([1, 2, 3]) * np.exp(-0.5 / 0.5))


def test_hybrid_update_residual_converges_to_y_minus_model():
    geom = Geometry.from_env(make_env(3), 30.0, M_NOM, 1.0)
    hyb = HybridTension(geom, 3, fc=1000.0, tau=0.5, dt=0.01)    # fc=1000：不濾波
    q = np.tile(np.array([0, 0, 1.0]), (3, 1))
    a = np.array([0.5, 0, 0])
    y = np.array([[1.0, 2, 3]] * 3)
    hyb.update(y, a, q)
    exp = y[0] - hyb.model_T(0, a[None])[0] * q[0]
    assert np.allclose(hyb.r[0], exp)


def test_est_eta_comp_uses_estimated_zeta_not_truth(monkeypatch):
    """事後對照：HYB-local / HYB 在 eta_comp='est' 下會呼叫 set_eta，且值 = 1 + 估計的 zeta（不是真值 k_true）。"""
    from smpc.eval.episode_p2c import run_episode_p2c
    for method in ("HYB-local", "HYB"):
        calls = []
        orig = SuspendedMPPITeam.set_eta
        monkeypatch.setattr(SuspendedMPPITeam, "set_eta", lambda self, i, v, _o=orig: (calls.append((i, float(v))), _o(self, i, v))[1])
        r = run_episode_p2c(method, "circle", 2001, "moderate", 1.0, {"fc": 8.0, "tau": 0.5}, eta_comp="est", T=1.0)
        monkeypatch.undo()
        k = np.array(r["k_true"])
        assert len(calls) >= 3 * 50, method                      # 每步每機都呼叫
        vals = np.array([v for _, v in calls])
        assert np.all(np.isfinite(vals))
        last = np.array([[v for i, v in calls if i == j][-1] for j in range(3)])
        assert not np.allclose(last, k, atol=1e-6), method        # 不是真值
        none_calls = []
        monkeypatch.setattr(SuspendedMPPITeam, "set_eta", lambda self, i, v: none_calls.append(v))
        run_episode_p2c(method, "circle", 2001, "moderate", 1.0, {"fc": 8.0, "tau": 0.5}, eta_comp="none", T=0.3)
        monkeypatch.undo()
        assert not none_calls                                     # none 不呼叫


def test_hyb_residual_uses_compensated_thrust_under_eta_comp(monkeypatch):
    """est 發散診斷：eta_comp!='none' 時，HYB 殘差通道的 y 必須相對 η̂·Uc 量（y_hyb = y + (1-η̂) f），否則 (k-1)f 被雙重修正。
    none 路徑必須原樣傳 y（不多不少）。"""
    from smpc.adapt.hybrid import HybridTension
    from smpc.eval.episode_p2c import run_episode_p2c
    seen = {}
    orig = HybridTension.update

    def spy(self, y, a_ref, q):
        seen.setdefault("ys", []).append(np.array(y))
        return orig(self, y, a_ref, q)
    for eta in ("none", "oracle"):
        seen.clear()
        monkeypatch.setattr(HybridTension, "update", spy)
        raw = []
        import smpc.eval.sensing as S
        o_d = S.SensorModel.delay_bundle

        def dbl(self, b, _o=o_d):
            raw.append((b["y"].copy(), b["f"].copy(), b.get("y_hyb")))
            return _o(self, b)
        monkeypatch.setattr(S.SensorModel, "delay_bundle", dbl)
        r = run_episode_p2c("HYB", "circle", 2001, "moderate", 1.0, {"fc": 8.0, "tau": 0.5}, eta_comp=eta, T=0.5)
        monkeypatch.undo()
        k = np.array(r["k_true"])
        if eta == "none":
            assert all(yh is None for _, _, yh in raw)
        else:
            y, f, yh = raw[-1]
            assert np.allclose(yh, y + (1 - k)[:, None] * f, atol=1e-9)     # oracle：η̂ = k_true
            assert not np.allclose(yh, y)


def test_ident_hook_does_not_change_result_and_records():
    import torch
    torch.set_num_threads(1)
    from smpc.eval.episode_p2c import run_episode_p2c
    a = run_episode_p2c("HYB-local", "circle", 2020, "moderate", 1.0, {"fc": 8.0, "tau": 6.0}, eta_comp="est", T=1.0)
    b = run_episode_p2c("HYB-local", "circle", 2020, "moderate", 1.0, {"fc": 8.0, "tau": 6.0}, eta_comp="est", T=1.0, ident=True)
    assert a["rmse"] == b["rmse"]
    d = b["ident"]
    assert d["zeta"].shape == (100, 3) and d["theta"].shape == (100, 3, 3) and d["Lam"].shape == (2, 3, 2, 2)


def test_locm_local_only_no_truth_no_residual_channel(monkeypatch):
    """LOC-M：本地 LocalDm 模型前饋 + 可選本地 ζ̂ 補償；不設 tension_fn（無殘差通道）、不用 payload 真值、est 時呼叫 set_eta。"""
    from smpc.eval.episode_p2c import run_episode_p2c
    th_calls, eta_calls, tf_calls = [], [], []
    o_th, o_eta = SuspendedMPPITeam.set_theta, SuspendedMPPITeam.set_eta
    monkeypatch.setattr(SuspendedMPPITeam, "set_theta", lambda self, i, v, _o=o_th: (th_calls.append((i, np.array(v))), _o(self, i, v))[1])
    monkeypatch.setattr(SuspendedMPPITeam, "set_eta", lambda self, i, v, _o=o_eta: (eta_calls.append((i, float(v))), _o(self, i, v))[1])
    monkeypatch.setattr(SuspendedMPPITeam, "set_tension_fn", lambda self, i, fn: tf_calls.append(i))
    r = run_episode_p2c("LOC-M", "circle", 2001, "moderate", 1.0, None, eta_comp="est", T=1.0)
    monkeypatch.undo()
    assert not tf_calls                                            # 無殘差通道
    assert len(th_calls) >= 3 * 50 and len(eta_calls) >= 3 * 50
    th_true = np.array(r["theta_true"])
    last = np.array([[v for i, v in th_calls if i == j][-1] for j in range(3)])
    assert not np.allclose(last[:, 0], th_true[0], atol=1e-6)      # 不是真值
    assert np.allclose(last[:, 1:], 0.0)                           # δc 固定為 0（LocalDm）
    assert not np.allclose([v for i, v in eta_calls][-3:], np.array(r["k_true"]), atol=1e-6)
    eta_calls.clear()
    monkeypatch.setattr(SuspendedMPPITeam, "set_eta", lambda self, i, v: eta_calls.append(v))
    run_episode_p2c("LOC-M", "circle", 2001, "moderate", 1.0, None, eta_comp="none", T=0.3)
    monkeypatch.undo()
    assert not eta_calls
