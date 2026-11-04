"""多機吊掛 spike 的物理 sanity 測試。門檻的來源見各測試註解與 results/p05_rope/spike.json。"""
import numpy as np
import pytest

from smpc.control.suspended_hover import SuspendedHover
from smpc.envs.multi_payload import MultiPayloadEnv

REF = np.array([0.0, 0.0, 1.0])


# (繩模型, 積分器)。spring+RK4 不在列：spike 實測 physics_dt>=2ms 時 payload 偏航會緩慢發散
# （見 results/p05_rope/spike.json），是已知限制，不是測試遺漏。
CONFIGS = [("tendon", "RK4"), ("tendon", "implicitfast"), ("spring", "implicitfast")]


def hover(model, integ="RK4", T=10.0, kick=False, **kw):
    env = MultiPayloadEnv(n=3, rope_model=model, integrator=integ, payload_mass=0.4, **kw)
    ctrl, tg = SuspendedHover(env), env.hover_positions()
    o = env.reset()
    drift, stretch, tmax = [], [], 0.0
    for k in range(int(T / env.dt)):
        if kick and k == 100:
            env.kick_payload([0, 0, 1.0])   # 向上衝量 -> 繩鬆弛 -> 再拉直
        o = env.step(ctrl(o["quad_states"], tg))
        assert np.isfinite(o["quad_states"]).all() and np.isfinite(o["payload_state"]).all()
        drift.append(np.linalg.norm(o["payload_state"][:3] - REF))
        stretch.append((o["rope_lengths"].max() - env.L) / env.L)
        tmax = max(tmax, o["tension_peak"].max())
    return np.array(drift), np.array(stretch), tmax, env


@pytest.mark.parametrize("model,integ", CONFIGS)
def test_hover_10s_no_drift(model, integ):
    drift, _, _, _ = hover(model, integ)
    assert drift.max() < 0.05, drift.max()
    assert drift[-100:].max() < 0.02  # 最後 1 s 已收斂，沒有發散趨勢


@pytest.mark.parametrize("model,integ", CONFIGS)
def test_rope_never_stretches_much(model, integ):
    # 先量再定：懸停最大伸長 ~0.06%，含 1 m/s 衝量造成的鬆弛->拉直也 < 0.8%（tendon 最大，
    # 見 spike.json）。門檻取 1%：是「繩近乎不可伸長」的工程判斷，不是貼著量測值。
    _, stretch, tmax, env = hover(model, integ, T=6.0, kick=True)
    assert stretch.max() < 0.01, stretch.max()
    assert tmax > 3.0  # 衝量確實造成 taut/slack 轉換（靜張力 1.5 N，峰值應明顯更大）


def test_static_feedforward_matches_truth():
    env = MultiPayloadEnv(n=3, payload_mass=0.4)
    T = env.static_tensions()
    assert np.allclose(T.sum() * np.cos(np.deg2rad(30)), 0.4 * 9.81, rtol=1e-6)


def test_models_agree_on_static_tension():
    # 兩種繩模型獨立實作，懸停張力真值應互相吻合（交叉驗證）。
    Ts = []
    for m, integ in [("tendon", "RK4"), ("spring", "implicitfast")]:
        _, _, _, env = hover(m, integ, T=6.0)
        Ts.append(env.rope_tension_truth)
    assert np.allclose(Ts[0], Ts[1], atol=0.05)
    assert np.allclose(Ts[0], 1.51, atol=0.05)


@pytest.mark.parametrize("n", [2, 4, 6])
def test_builds_for_other_n(n):
    env = MultiPayloadEnv(n=n)
    o = env.reset()
    assert o["quad_states"].shape == (n, 13) and o["rope_lengths"].shape == (n,)
    assert np.allclose(o["rope_lengths"], env.L)
