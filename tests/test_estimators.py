"""P2a-2 估計器測試：CEN = 批次最小平方、complete 圖收斂到 CEN、NOM 輸出 0、閉迴路煙霧測試。"""
import numpy as np
import torch

from smpc.adapt.estimators import NS, PRIOR_STD, ZETA_STD, Geometry, SharedEstimator, tension_map, tension_model
from smpc.control.suspended_mppi import MPPIConfig, SuspendedMPPITeam
from smpc.control.trajectories import Traj
from smpc.eval.episode import M_NOM, make_env
from smpc.eval.episode_est import run_episode_est


def _geom():
    return Geometry.from_env(make_env(3))


def _synthetic(geom, steps=300, seed=0, th_true=(0.2, 0.02, -0.01), eta=(1.0, 0.9, 1.0)):
    """合成量測：y_i = T_i(θ) q_i + (η_i - 1) f_i + noise，沿圓軌跡。"""
    rng = np.random.default_rng(seed)
    traj = Traj("circle")
    N = geom.n
    th_true = np.asarray(th_true)
    data = []
    for k in range(steps):
        t = 0.01 * k + 2.0
        a_ref = traj.ref(t)[2][0]
        q = -(geom.offsets - geom.attach_body)
        q = q / np.linalg.norm(q, axis=1, keepdims=True) + 0.01 * rng.standard_normal((N, 3))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        f = np.stack([[0.3 * np.sin(0.1 * k + i), 0.2 * np.cos(0.07 * k), 9.0 + i] for i in range(N)])
        y = np.stack([tension_model(geom, th_true, a_ref, i)[0] * q[i] + (eta[i] - 1) * f[i] for i in range(N)])
        y = y + 0.15 * rng.standard_normal((N, 3))
        data.append((a_ref, q, f, y))
    return data


def test_tension_map_matches_controller():
    env = make_env(3)
    team = SuspendedMPPITeam(env, Traj("hover"), MPPIConfig(K=16, H=4), theta_hat=np.zeros(3), m_nom=M_NOM)
    geom = Geometry.from_env(env, MPPIConfig().cone_deg)
    for c in [(0, 0), (0.03, -0.02)]:
        assert np.allclose(tension_map(geom, c), team._tension_maps(np.array(c)), atol=1e-12)


def test_nom_outputs_zero():
    geom = _geom()
    est = SharedEstimator("NOM", geom)
    for a_ref, q, f, y in _synthetic(geom, 50):
        est.update(a_ref, q, f, y)
    assert np.all(est.theta == 0)


def test_cen_equals_batch_ls():
    """relin=False 時 H 與估計無關，遞迴 CEN 必須等於加權嶺迴歸（遺忘 λ<1，先驗不遺忘）。"""
    geom = _geom()
    lam, N = 0.99, geom.n
    est = SharedEstimator("CEN", geom, lam=lam, relin=False)
    data = _synthetic(geom, 200)
    Hs, ys, ws = [], [], []
    for a_ref, q, f, y in data:
        est.update(a_ref, q, f, y)
        H, yp = est._linearize(a_ref, q, y)
        Hc = np.zeros((3 * N, NS + N))
        for i in range(N):
            Hc[3 * i:3 * i + 3, :NS] = H[i]
            Hc[3 * i:3 * i + 3, NS + i] = f[i]
        Hs.append(Hc); ys.append(yp.reshape(-1))
    n = len(data)
    L0 = np.diag(np.concatenate([1 / PRIOR_STD**2, np.full(N, 1 / ZETA_STD**2)]))
    Lam, xi = L0.copy(), np.zeros(NS + N)
    for k in range(n):
        w = lam ** (n - 1 - k) / est.sigma_f**2
        Lam += w * Hs[k].T @ Hs[k]
        xi += w * Hs[k].T @ ys[k]
    ref = np.linalg.solve(Lam, xi)
    assert np.allclose(est.theta[0], ref[:NS], atol=1e-9)
    assert np.allclose(est.zeta, ref[NS:], atol=1e-9)
    assert np.allclose(est.theta, est.theta[0])           # 每機用同一個 θ̂


def _run(kind, data, **kw):
    est = SharedEstimator(kind, _geom(), relin=False, **kw)
    for a_ref, q, f, y in data:
        est.update(a_ref, q, f, y)
    return est


def test_complete_graph_consensus_converges_to_cen():
    data = _synthetic(_geom(), 200)
    cen = _run("CEN", data)
    con1 = _run("CON", data, graph="complete", iters=1)      # complete 圖 Metropolis 權重 1 次迭代即精確平均
    assert np.allclose(con1.theta, cen.theta, atol=1e-8)
    assert np.allclose(con1.zeta, cen.zeta, atol=1e-8)
    # path 圖：迭代越多越接近 CEN，且 ≥ 60 次後等於 CEN（Metropolis path 第二大特徵值 2/3）
    e = [np.abs(_run("CON", data, graph="path", iters=it).theta - cen.theta).max() for it in (1, 2, 5, 60)]
    assert e[0] > e[1] > e[2] >= e[3] and e[3] < 1e-8, e
    ind = _run("IND", data)
    assert np.abs(ind.theta - cen.theta).max() > e[1]  # IND 與 CEN 確實不同（不通訊 != 融合）


def test_slack_weight_zero_equals_skip_data():
    geom = _geom()
    data = _synthetic(geom, 80)
    a = SharedEstimator("IND", geom, relin=False)
    b = SharedEstimator("IND", geom, relin=False)
    for k, (a_ref, q, f, y) in enumerate(data):
        act = np.array([k % 2 == 0, True, True])
        a.update(a_ref, q, f, y, act)
        y2 = y.copy(); y2[0] += 100.0 * (not act[0])           # 被標 slack 的資料就算爆掉也不能影響
        b.update(a_ref, q, f, y2, act)
    assert np.allclose(a.theta, b.theta)


def test_closed_loop_smoke_one_episode():
    torch.set_num_threads(2)
    for m in ("IND", "CON", "CEN"):
        r = run_episode_est(m, "circle", 2000, T=2.5, cfg=MPPIConfig.load("configs/mppi_frozen.json", K=128))
        assert r["finite"] and r["max_err"] < 0.3, (m, r["max_err"])
        assert np.isfinite(r["dm_err@2"]) and r["t_est_ms"] < 20
