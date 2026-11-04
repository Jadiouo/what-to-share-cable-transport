"""P2a-2 估計器：NOM / IND / CON / CEN / ORACLE（共享量 θ_s = (δm, δc_x, δc_y)，本地量 ζ_i = η_i - 1）。

量測模型（PLAN R1：沒有張力計）。機 i 的加速度計量到比力 m a_imu = η_i f_i + T_i q_i（f_i = 自己送出的推力向量，
q_i = 自己的繩方向，機 -> 掛點）。定義「力殘差」
        y_i := m a_imu,i - f_i  =  T_i q_i + ζ_i f_i  + noise            （ζ_i = η_i - 1）
回歸模型（名義幾何 + 參考軌跡，不含任何 payload 真值）：
        y_i ≈ T_i^model(θ) q̃_i + ζ_i f_i ,   T^model(θ) = P(c) (m0 + δm) (a_ref + g e_z)
P(c) 是控制器前饋用的同一個張力映射（名義編隊幾何，力權重 1、力矩權重 w_moment 的 lstsq）。
沿參考軌跡重線性化（EKF 式，與 P0.5c 的 `relin` 同）：在目前的 θ̂ 處
        H_i = q̃_i (∂T_i/∂θ)(θ̂)   (3x3，秩 1)；  y'_i = y_i - (T_i(θ̂) - (∂T_i/∂θ) θ̂) q̃_i = H_i θ + ζ_i f_i
即每機只看到一個純量張力偏差（沿 q̃_i），θ_s 的三維被壓成一維——這是 H1 的可辨識性來源。

各估計器（皆為資訊形式 RLS，遺忘只遺忘資料、不遺忘先驗，見 rls.py）：
  NOM     θ̂ = 0。
  IND     每機本地 RLS，參數 [θ(3), ζ_i]，不通訊；θ̂_i 由自己的資訊解出，用於自己。
  CON     每機同 IND 的本地 RLS；每步把本地資訊對 ζ_i 做 Schur 補得 θ 的邊際 (M_i, v_i)，用 Metropolis 權重做
          `iters` 次平均，再 Λ_f = N·mean - (N-1)Λ0（資料加總、先驗只算一次）；本地 ζ̂_i 用 conditional_local。
          這是 P0.5c 的「非遞迴資訊 consensus」：本地狀態只吃自己的資料，融合結果不回灌，所以不會重複計算；
          它是 HCMCI 的簡化（沒有 CM/CI 混合權重），不是原論文演算法（P4 前要核對）。
  CEN     所有機的資訊相加：單一 RLS，參數 [θ(3), ζ_1..ζ_N]；每機用同一個 θ̂。
  ORACLE  θ̂ = 真值（由呼叫端給）。
slack：呼叫端用幾何判斷後傳 active[i]=False，該機這一步的資訊權重為 0（仍套用遺忘）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smpc.adapt.rls import InfoRLS, conditional_local, consensus, metropolis_weights, schur_marginal

PRIOR_STD = np.array([0.5, 0.1, 0.1])    # P0.5c：δm 0.5 kg、δc 0.1 m
ZETA_STD = 0.3                            # P0.5c
NS = 3
KINDS = ("NOM", "IND", "CON", "CEN", "ORACLE")


@dataclass
class Geometry:
    attach_body: np.ndarray    # (N,3) 掛點（payload 座標，名義）
    offsets: np.ndarray        # (N,3) 名義編隊：quad 位置 - payload 位置
    m_nom: float = 1.0
    g: float = 9.81
    w_moment: float = 1.0

    @staticmethod
    def from_env(env, cone_deg=30.0, m_nom=1.0, w_moment=1.0):
        return Geometry(np.array(env.attach_body, float), env.hover_positions(np.zeros(3), cone_deg),
                        m_nom, env.p.g, w_moment)

    @property
    def n(self):
        return len(self.attach_body)


def tension_map(geom: Geometry, c) -> np.ndarray:
    """名義幾何、重心偏移 c=(cx,cy) 下，payload 力需求 (3,) -> 各繩張力的線性映射 (N,3)。與控制器前饋相同的定義。"""
    wq = geom.offsets - geom.attach_body
    wq = wq / np.linalg.norm(wq, axis=1, keepdims=True)
    cv = np.array([c[0], c[1], 0.0])
    A = np.zeros((6, geom.n))
    A[:3] = wq.T
    A[3:] = np.cross(geom.attach_body - cv, wq).T
    W = np.diag([1.0, 1.0, 1.0] + [geom.w_moment] * 3)
    return np.linalg.pinv(W @ A)[:, :3]


def tension_model(geom: Geometry, theta, a_ref, i=None, h=1e-3):
    """T^model(θ)（N,）與 ∂T/∂θ（N,3）。δm 的導數解析；δc 用前向差分。i 不為 None 時只回傳第 i 條繩（純量, (3,)）。"""
    th = np.asarray(theta, float)
    a_up = np.asarray(a_ref, float) + np.array([0, 0, geom.g])
    P0 = tension_map(geom, th[1:3])
    f_req = (geom.m_nom + th[0]) * a_up
    T = P0 @ f_req
    J = np.zeros((geom.n, 3))
    J[:, 0] = P0 @ a_up
    for k in range(2):
        c2 = th[1:3].copy()
        c2[k] += h
        J[:, 1 + k] = (tension_map(geom, c2) @ f_req - T) / h
    if i is not None:
        return T[i], J[i]
    return T, J


class SharedEstimator:
    """N 機的共享量估計器。update() 每控制步呼叫一次；theta (N,3) 是各機目前採用的 θ̂_s。"""

    def __init__(self, kind: str, geom: Geometry, lam: float = 0.9995, sigma_f: float = 0.15, graph: str = "path",
                 iters: int = 2, relin: bool = True, theta_true=None, prior_std=None, zeta_std: float = ZETA_STD):
        assert kind in KINDS
        self.kind, self.geom, self.lam, self.relin = kind, geom, lam, relin
        self.n = N = geom.n
        prior_std = PRIOR_STD if prior_std is None else np.asarray(prior_std, float)
        self.sigma_f = sigma_f
        self.iters = iters
        self.theta = np.zeros((N, NS))
        self.zeta = np.zeros(N)
        self.M = np.broadcast_to(np.diag(1 / prior_std**2), (N, NS, NS)).copy()   # θ 邊際資訊（NEES 用）
        self.L0s = np.diag(1 / prior_std**2)
        if kind == "ORACLE":
            assert theta_true is not None
            self.theta = np.tile(np.asarray(theta_true, float), (N, 1))
        if graph == "path":
            adj = np.eye(N, k=1) + np.eye(N, k=-1)
        elif graph == "complete":
            adj = np.ones((N, N)) - np.eye(N)
        else:
            raise ValueError(graph)
        self.W = metropolis_weights(adj)
        if kind in ("IND", "CON"):
            P0 = np.diag(np.concatenate([1 / prior_std**2, [1 / zeta_std**2]]))
            self.rls = InfoRLS(np.broadcast_to(P0, (N, NS + 1, NS + 1)).copy(), np.zeros((N, NS + 1)), lam)
        elif kind == "CEN":
            Pc = np.diag(np.concatenate([1 / prior_std**2, np.full(N, 1 / zeta_std**2)]))
            self.rls = InfoRLS(Pc, np.zeros(NS + N), lam)

    # ------------------------------------------------------------------
    def _linearize(self, a_ref, q, y):
        """每機的 (H_i (3x3), y'_i (3,))：在各機目前線性化點處。"""
        N = self.n
        H = np.zeros((N, 3, NS))
        yp = y.copy()
        for i in range(N):
            th_lin = self.theta[i] if self.relin else np.zeros(NS)
            Ti, Ji = tension_model(self.geom, th_lin, a_ref, i)
            H[i] = np.outer(q[i], Ji)
            yp[i] = y[i] - (Ti - Ji @ th_lin) * q[i]
        return H, yp

    def update(self, a_ref, q, f, y, active=None):
        """a_ref (3,)：此步 payload 參考加速度；q (N,3)：量到的繩方向（機->掛點）；f (N,3)：各機送出的推力向量；
        y (N,3)：力殘差 m a_imu - f；active (N,) bool：False = slack，資訊權重 0。"""
        N = self.n
        if self.kind in ("NOM", "ORACLE"):
            return
        active = np.ones(N, bool) if active is None else np.asarray(active, bool)
        H, yp = self._linearize(np.asarray(a_ref, float), q, y)
        w = active / self.sigma_f**2
        if self.kind == "CEN":
            Hc = np.zeros((3 * N, NS + N))
            for i in range(N):
                Hc[3 * i:3 * i + 3, :NS] = H[i]
                Hc[3 * i:3 * i + 3, NS + i] = f[i]
            Rinv = np.diag(np.repeat(w, 3))
            self.rls.update(Hc, yp.reshape(-1), Rinv)
            Lam = self.rls.Lambda
            est = self.rls.estimate
            Mc = Lam[:NS, :NS] - Lam[:NS, NS:] @ np.linalg.solve(Lam[NS:, NS:], Lam[NS:, :NS])
            self.theta = np.tile(est[:NS], (N, 1))
            self.zeta = est[NS:].copy()
            self.M = np.tile(Mc, (N, 1, 1))
            return
        Hf = np.concatenate([H, f[..., None]], -1)                 # (N,3,4)
        Rinv = w[:, None, None] * np.eye(3)[None]
        self.rls.update(Hf, yp, Rinv)
        M, v = schur_marginal(self.rls.Lambda, self.rls.xi, NS)
        if self.kind == "CON":
            Mb = consensus(M, self.W, self.iters, -3)
            vb = consensus(v, self.W, self.iters, -2)
            M, v = N * Mb - (N - 1) * self.L0s, N * vb
        th = np.linalg.solve(M, v[..., None])[..., 0]
        self.theta = th
        self.zeta = conditional_local(self.rls.Lambda, self.rls.xi, th, NS)[..., 0]
        self.M = M

    def nees(self, theta_true):
        e = self.theta - np.asarray(theta_true, float)
        return np.einsum("ni,nij,nj->n", e, self.M, e) / NS
