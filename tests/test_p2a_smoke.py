"""P2a-1 冒煙測試：N=3 ORACLE 懸停 3 s 不發散；成本向量化版與逐步版一致；θ̂ 介面可替換。"""
import numpy as np
import torch

from smpc.control.suspended_mppi import MPPIConfig, SuspendedMPPITeam
from smpc.control.trajectories import Traj
from smpc.eval.episode import M_NOM, make_env, rope_dirs, run_episode


def test_oracle_hover_3s_no_divergence():
    torch.set_num_threads(2)
    r = run_episode("mppi", "hover", seed=0, T=3.0, cfg=MPPIConfig(K=256, lam=1.0, sigma=0.4))
    assert r["finite"] and r["max_err"] < 0.15, r
    assert r["final_err"] < 0.05, r


def test_traj_cost_matches_stepwise_cost():
    env = make_env(3)
    team = SuspendedMPPITeam(env, Traj("circle"), MPPIConfig(K=64, H=10), theta_hat=np.zeros(3), m_nom=M_NOM)
    o = env.reset()
    a = team.agents[0]
    a.act(0.5, o["quad_states"][0], rope_dirs(env)[0])      # 填好 _pq/_d/_a_ff 等
    K, H = 16, a.cfg.H
    g = torch.Generator().manual_seed(1)
    x = torch.as_tensor(np.r_[o["quad_states"][0][:3], o["quad_states"][0][7:10], np.zeros(6)],
                        dtype=torch.float32).expand(K, 12).clone()
    U = torch.randn(K, H, 3, generator=g)
    xs, S = [], torch.zeros(K)
    for k in range(H):
        x = a._dyn(x, U[:, k], k)
        xs.append(x)
        S = S + a._cost(x, U[:, k], k)
    assert torch.allclose(S, a._tcost(torch.stack(xs, 1), U), rtol=1e-4, atol=1e-3)


def test_theta_interface_changes_feedforward_only():
    """換估計器 = 呼叫 set_theta；控制器其餘不動。δm 增加 -> 前饋張力增加。"""
    env = make_env(3)
    team = SuspendedMPPITeam(env, Traj("hover"), MPPIConfig(K=32, H=5), theta_hat=np.zeros(3), m_nom=M_NOM)
    o = env.reset()
    team.act(0.0, o["quad_states"], rope_dirs(env))
    T0 = team.feedforward_tension().copy()
    team.set_theta(0, np.array([0.3, 0.0, 0.0]))
    team.act(0.0, o["quad_states"], rope_dirs(env))
    T1 = team.feedforward_tension()
    assert T1[0] > T0[0] * 1.2 and np.isclose(T1[1], T0[1])
