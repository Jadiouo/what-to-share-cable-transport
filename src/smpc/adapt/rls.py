"""資訊形式遞迴最小平方（RLS）與 consensus 小工具。

狀態是資訊矩陣 Lambda 與資訊向量 xi（estimate = Lambda^{-1} xi）。資訊形式的好處：
多機融合就是把 (Lambda, xi) 相加或平均，不必處理共變異數的互相關。

所有函式支援任意前導 batch 維度（例如 seeds x agents），形狀為 (..., d, d) / (..., d)。

遺忘方式（重要）：只遺忘「資料」，不遺忘先驗。
    Lambda <- lam * Lambda + (1 - lam) * Lambda0 + H' R^-1 H
    xi     <- lam * xi     + (1 - lam) * xi0     + H' R^-1 y
這樣沒有激勵的方向會慢慢回到先驗，而不是讓 Lambda 衰減到奇異。lam=1 時退化成標準 RLS。
"""
from __future__ import annotations

import numpy as np


class InfoRLS:
    """資訊形式 RLS（含遺忘因子與先驗）。

    Lambda0: (..., d, d) 先驗資訊（= 1/先驗變異數）；mu0: (..., d) 先驗均值。
    """

    def __init__(self, Lambda0: np.ndarray, mu0: np.ndarray, lam: float = 1.0):
        self.Lambda0 = np.array(Lambda0, float)
        self.xi0 = np.einsum("...ij,...j->...i", self.Lambda0, np.array(mu0, float))
        self.lam = float(lam)
        self.Lambda = self.Lambda0.copy()
        self.xi = self.xi0.copy()

    def update(self, H: np.ndarray, y: np.ndarray, Rinv=1.0):
        """H: (..., m, d)，y: (..., m)，Rinv: 純量（白雜訊 1/sigma^2）或 (..., m, m)。"""
        if np.ndim(Rinv) == 0:
            HtR = H * Rinv
        else:
            HtR = np.einsum("...mk,...mn->...kn", Rinv, H)  # R^-1 H
        Ht = np.swapaxes(H, -1, -2)
        a = self.lam
        if np.ndim(Rinv) == 0:
            dL = np.einsum("...mi,...mj->...ij", H, HtR)
            dx = np.einsum("...mi,...m->...i", HtR, y)
        else:
            dL = np.einsum("...im,...mj->...ij", Ht, HtR)
            dx = np.einsum("...im,...m->...i", Ht, np.einsum("...mn,...n->...m", Rinv, y))
        self.Lambda = a * self.Lambda + (1 - a) * self.Lambda0 + dL
        self.xi = a * self.xi + (1 - a) * self.xi0 + dx

    @property
    def estimate(self) -> np.ndarray:
        return np.linalg.solve(self.Lambda, self.xi[..., None])[..., 0]

    @property
    def covariance(self) -> np.ndarray:
        return np.linalg.inv(self.Lambda)


def schur_marginal(Lam: np.ndarray, xi: np.ndarray, ns: int):
    """把資訊 (Lam, xi) 的後 d-ns 個座標（本地量）邊際化掉，回傳前 ns 個座標（共享量）的資訊。

    M = Lss - Lsl Lll^-1 Lls ；v = xs - Lsl Lll^-1 xl。對高斯，這就是共享量的邊際資訊。
    """
    Lss, Lsl = Lam[..., :ns, :ns], Lam[..., :ns, ns:]
    Lll = Lam[..., ns:, ns:]
    xs, xl = xi[..., :ns], xi[..., ns:]
    K = np.linalg.solve(Lll, np.swapaxes(Lsl, -1, -2))  # Lll^-1 Lls
    M = Lss - np.einsum("...sl,...lt->...st", Lsl, K)
    sol = np.linalg.solve(Lll, xl[..., None])[..., 0]
    v = xs - np.einsum("...sl,...l->...s", Lsl, sol)
    return M, v


def conditional_local(Lam: np.ndarray, xi: np.ndarray, theta_s: np.ndarray, ns: int):
    """給定共享量估計 theta_s，本地量的條件估計：eta = Lll^-1 (xl - Lls theta_s)。"""
    Lls = Lam[..., ns:, :ns]
    Lll = Lam[..., ns:, ns:]
    rhs = xi[..., ns:] - np.einsum("...ls,...s->...l", Lls, theta_s)
    return np.linalg.solve(Lll, rhs[..., None])[..., 0]


def metropolis_weights(adj: np.ndarray) -> np.ndarray:
    """Metropolis–Hastings 權重（雙隨機、對稱）。adj: (N,N) 0/1 無自環鄰接矩陣。"""
    adj = np.array(adj, float)
    np.fill_diagonal(adj, 0)
    deg = adj.sum(1)
    N = len(adj)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if adj[i, j]:
                W[i, j] = 1.0 / (1 + max(deg[i], deg[j]))
    W[np.diag_indices(N)] = 1.0 - W.sum(1)
    return W


def consensus(X: np.ndarray, W: np.ndarray, iters: int = 1, agent_axis: int = -3) -> np.ndarray:
    """對 agent 軸做 iters 次 X <- W X。X 形狀 (..., N, a, b) 或 (..., N, a)（agent_axis 指 N 的位置）。"""
    X = np.moveaxis(X, agent_axis, 0)
    for _ in range(iters):
        X = np.tensordot(W, X, axes=(1, 0))
    return np.moveaxis(X, 0, agent_axis)
