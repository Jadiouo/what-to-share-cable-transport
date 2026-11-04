"""模型無關的批次 MPPI（PyTorch CPU）。

介面（與派工單的差異：多傳一個步數索引 k，讓參考軌跡／繩力前饋這類「隨時間變、與狀態無關」的
量可以進模型而不必每步重建函式）：
    dynamics(x, u, k) -> x_next      x: (K,nx)，u: (K,nu)，k: 0..H-1
    cost(x_next, u, k) -> (K,)       階段成本（x_next 是 u 作用後的狀態）
    terminal(x_H) -> (K,)            可選
    traj_cost(X, U) -> (K,)          可選：整條軌跡一次算完（X: (K,H,nx) 為 x_1..x_H，U: (K,H,nu)），
                                     給了就不逐步呼叫 cost；結果相同、只是少掉 H 次 Python 迴圈。
模型若不需要 k，忽略即可。控制變數 U (H,nu) 是被最佳化的量；若想「以前饋為中心」，
由 dynamics 內部把前饋加在 u 上（殘差參數化），MPPI 本身不知道。

演算法（Williams et al. 2017 的資訊理論版，無額外控制成本項，控制成本由 cost 自訂）：
    ε_k ~ N(0, Σ)；第 0 個樣本固定 ε=0（保證名義序列在候選裡）
    S_j = Σ_k cost；w_j ∝ exp(-(S_j - min S)/λ)；U ← U + Σ_j w_j ε_j
warm start：每次 plan 結束後把 U 左移一格（最後一格複製），下一次從那裡開始。
"""
from __future__ import annotations

import numpy as np
import torch


class MPPI:
    def __init__(self, dynamics, cost, nx: int, nu: int, K: int = 512, H: int = 20,
                 lam: float = 1.0, sigma=1.0, noise_corr: float = 0.0, terminal=None, traj_cost=None, u_min=None, u_max=None,
                 seed: int = 0, dtype=torch.float32):
        self.f, self.c, self.term, self.tc = dynamics, cost, terminal, traj_cost
        self.nx, self.nu, self.K, self.H, self.lam = nx, nu, K, H, float(lam)
        self.dtype = dtype
        self.rho = float(noise_corr)     # 取樣噪聲的時間相關係數 ρ（AR(1)，邊際標準差仍是 σ）；0 = 獨立
        self.sigma = torch.as_tensor(np.broadcast_to(np.asarray(sigma, float), (nu,)).copy(), dtype=dtype)
        self.u_min = None if u_min is None else torch.as_tensor(np.broadcast_to(u_min, (nu,)).copy(), dtype=dtype)
        self.u_max = None if u_max is None else torch.as_tensor(np.broadcast_to(u_max, (nu,)).copy(), dtype=dtype)
        self.gen = torch.Generator().manual_seed(seed)
        self.U = torch.zeros(H, nu, dtype=dtype)
        self.last = {}

    def reset(self, U0=None):
        self.U = torch.zeros(self.H, self.nu, dtype=self.dtype) if U0 is None \
            else torch.as_tensor(U0, dtype=self.dtype).clone()

    def _clip(self, u):
        if self.u_min is not None:
            u = torch.maximum(u, self.u_min)
        if self.u_max is not None:
            u = torch.minimum(u, self.u_max)
        return u

    @torch.no_grad()
    def plan(self, x0) -> np.ndarray:
        """回傳第一步控制 (nu,)，並更新（平移後的）內部序列。"""
        K, H, nu = self.K, self.H, self.nu
        eps = torch.randn(K, H, nu, generator=self.gen, dtype=self.dtype)
        if self.rho > 0:
            r, q = self.rho, (1 - self.rho ** 2) ** 0.5
            for k in range(1, H):
                eps[:, k] = r * eps[:, k - 1] + q * eps[:, k]
        eps = eps * self.sigma
        eps[0] = 0.0
        Us = self._clip(self.U[None] + eps)
        eps = Us - self.U[None]                      # 夾限後的實際擾動，更新才與成本一致
        x = torch.as_tensor(x0, dtype=self.dtype).expand(K, self.nx).clone()
        S = torch.zeros(K, dtype=self.dtype)
        if self.tc is not None:
            xs = []
            for k in range(H):
                x = self.f(x, Us[:, k], k)
                xs.append(x)
            S = S + self.tc(torch.stack(xs, 1), Us)
        else:
            for k in range(H):
                u = Us[:, k]
                x = self.f(x, u, k)
                S = S + self.c(x, u, k)
        if self.term is not None:
            S = S + self.term(x)
        S = torch.nan_to_num(S, nan=1e12, posinf=1e12)
        w = torch.exp(-(S - S.min()) / self.lam)
        w = w / w.sum()
        self.U = self._clip(self.U + (w[:, None, None] * eps).sum(0))
        u0 = self.U[0].clone()
        self.last = dict(cost_min=float(S.min()), cost_nom=float(S[0]), ess=float(1.0 / (w ** 2).sum()))
        self.U = torch.cat([self.U[1:], self.U[-1:]], 0)      # 時間平移 warm start
        return u0.numpy()
