"""單機 quadrotor 物理 sanity 測試。門檻照派工單，不為了通過而放寬。"""
import numpy as np
import pytest

from smpc.control.pd import GeometricPD
from smpc.envs.params import QuadParams
from smpc.envs.quadrotor import QuadrotorEnv

P = QuadParams()


def fly(env, ctrl, target, T, state0=None, ext=None):
    """跑 T 秒，回傳 (state 軌跡, 是否曾飽和)。"""
    s = env.reset(state0)
    traj, sat = [s], False
    for _ in range(int(round(T / env.dt))):
        s = env.step(ctrl(s, target))
        sat |= ctrl.saturated
        traj.append(s)
    return np.array(traj), sat


def test_free_fall_matches_g():
    env = QuadrotorEnv(P)
    env.reset()
    env.step(np.zeros(4))
    v0 = env.state[9]
    for _ in range(49):
        env.step(np.zeros(4))
    a = (env.state[9] - v0) / (49 * env.dt)
    assert abs(a + P.g) / P.g < 0.01, a


def test_hover_thrust_no_drift():
    env = QuadrotorEnv(P)
    env.reset()
    u = np.full(4, P.mass * P.g / 4)
    for _ in range(int(2 / env.dt)):
        env.step(u)
    assert np.linalg.norm(env.state[7:10]) < 0.01  # < 1 cm/s
    assert np.linalg.norm(env.state[10:13]) < 1e-6


def test_allocation_signs():
    """mixer 的符號必須與 MJCF 一致：給 tau_x 力矩，角加速度應 = tau/I 同號。"""
    env = QuadrotorEnv(P)
    for axis in range(3):
        env.reset()
        w = np.array([P.mass * P.g, 0, 0, 0.0])
        w[1 + axis] = 0.002
        u = np.linalg.inv(P.allocation) @ w
        env.step(u)
        alpha = env.state[10 + axis] / env.dt
        assert abs(alpha - 0.002 / P.inertia[axis]) / (0.002 / P.inertia[axis]) < 0.02, (axis, alpha)


def test_pd_hover_from_offset():
    env, pd = QuadrotorEnv(P), GeometricPD(P)
    s0 = np.array([0.2, 0, 1.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    traj, _ = fly(env, pd, np.array([0, 0, 1.0]), 5.0, s0)
    assert np.linalg.norm(traj[-1, :3] - [0, 0, 1.0]) < 0.02


def test_pd_reach_target_1m():
    env, pd = QuadrotorEnv(P), GeometricPD(P)
    target = np.array([1.0, 0, 1.0])
    traj, sat = fly(env, pd, target, 5.0)
    assert np.linalg.norm(traj[-1, :3] - target) < 0.05
    assert np.all(np.isfinite(traj)) and np.abs(traj[:, :3]).max() < 3.0  # 不發散
    assert not sat  # 1 m 階躍不該需要飽和


@pytest.mark.xfail(strict=True, reason=(
    "實測：純 PD 無積分，單馬達 η=0.7 穩定但穩態位置誤差約 0.32 m（水平 ~0.30、掉高 ~0.10），"
    "超過 10 cm 門檻。這是 P2a 要用適應估計補的缺口；未調參數去過。"))
def test_single_motor_eta_07_hover():
    """記錄能力：純 PD（無積分/無適應）在一顆馬達 η=0.7 時能否懸停。誤差門檻 10 cm。"""
    env, pd = QuadrotorEnv(P), GeometricPD(P)
    target = np.array([0, 0, 1.0])
    env.reset()
    env.eta[0] = 0.7
    s = env.state
    for _ in range(int(5 / env.dt)):
        s = env.step(pd(s, target))
    err = np.linalg.norm(s[:3] - target)
    print(f"eta=0.7 final position error: {err:.4f} m")
    assert err < 0.10, err


def test_external_force_hook():
    env = QuadrotorEnv(P)
    env.reset()
    env.ext_force = np.array([1.0, 0, 0])
    u = np.full(4, P.mass * P.g / 4)
    n = 10
    for _ in range(n):
        env.step(u)
    ax = env.state[7] / (n * env.dt)
    assert abs(ax - 1.0 / P.mass) < 0.02 * (1.0 / P.mass), ax
    env.set_mass(2.0)  # 質量鉤子：同樣外力下加速度減半
    env.data.qvel[:] = 0
    for _ in range(n):
        env.step(2 * u)
    assert abs(env.state[7] / (n * env.dt) - 0.5) < 0.02


def test_single_motor_eta_07_bounded():
    """η=0.7 時純 PD 不發散、不飽和失控：誤差有界（< 0.5 m）且靜止。"""
    env, pd = QuadrotorEnv(P), GeometricPD(P)
    env.reset()
    env.eta[0] = 0.7
    s = env.state
    for _ in range(int(5 / env.dt)):
        s = env.step(pd(s, np.array([0, 0, 1.0])))
    assert np.linalg.norm(s[:3] - [0, 0, 1.0]) < 0.5
    assert np.linalg.norm(s[7:10]) < 0.05
