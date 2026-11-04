"""MPPI 單元測試：1D 雙積分器收斂、warm start、可重現性。"""
import numpy as np
import torch

from smpc.control.mppi import MPPI

DT = 0.05
GOAL = 1.0


def dyn(x, u, k):
    p, v = x[:, 0], x[:, 1]
    a = u[:, 0]
    return torch.stack([p + v * DT + 0.5 * a * DT ** 2, v + a * DT], 1)


def cost(x, u, k):
    return 10 * (x[:, 0] - GOAL) ** 2 + 1.0 * x[:, 1] ** 2 + 0.01 * u[:, 0] ** 2


def make(seed=0, **kw):
    return MPPI(dyn, cost, nx=2, nu=1, K=256, H=20, lam=1.0, sigma=1.0, u_min=-3.0, u_max=3.0,
                seed=seed, **kw)


def run(m, steps=120):
    x = np.zeros(2)
    for _ in range(steps):
        a = m.plan(x)[0]
        x = np.array([x[0] + x[1] * DT + 0.5 * a * DT ** 2, x[1] + a * DT])
    return x


def test_double_integrator_converges():
    torch.set_num_threads(2)
    x = run(make())
    assert abs(x[0] - GOAL) < 0.05 and abs(x[1]) < 0.1, x


def test_control_bounds_respected():
    m = make()
    for _ in range(5):
        u = m.plan(np.zeros(2))
        assert -3.0 - 1e-6 <= u[0] <= 3.0 + 1e-6
    assert (m.U.abs() <= 3.0 + 1e-6).all()


def test_warm_start_shifts_sequence():
    m = make()
    m.plan(np.zeros(2))
    U_after = m.U.clone()
    # plan 結束時 U 已左移：U[-1] 複製 U[-2]
    assert torch.allclose(U_after[-1], U_after[-2])


def test_same_seed_reproducible():
    a = [make(seed=3).plan(np.zeros(2)) for _ in range(2)]
    assert np.allclose(a[0], a[1])
