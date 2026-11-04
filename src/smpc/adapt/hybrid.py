"""P2c 混合法：共享（或本地）模型預測未來張力 + 本地 DOB 修正當下偏差。

  F̂_i(t+s) = T_model,i(t+s) q_i + r_i(t) exp(-s/τ)           s = 預測時域內往前的時間
  r_i       = LPF_fc[ y_i - T_model,i(t) q_i ]                  （向量；y_i 是力殘差 = DOB 的原始輸出，含推力係數等所有未建模的力）
  T_model,i = max( P_i(ĉ) (m0 + δm̂)(a_ref + g e_z), 0 )          與控制器內建前饋同一個式子（r=0 時兩者逐位相同）
θ̂ 來源：HYB = CEN-ATT 的共享 θ̂（δm, δc）；HYB-local = IND-δm 的本地 θ̂（δm, 0）。
"""
from __future__ import annotations

import numpy as np

from smpc.adapt.estimators import Geometry, tension_map


class HybridTension:
    def __init__(self, geom: Geometry, n, fc, tau, dt=0.01):
        self.geom, self.n, self.tau = geom, n, float(tau)
        self.alpha = 1 - np.exp(-2 * np.pi * fc * dt)
        self.theta = np.zeros((n, 3))
        self.r = np.zeros((n, 3))
        self.init = False
        self._P = [tension_map(geom, np.zeros(2))[i] for i in range(n)]

    def set_theta(self, i, th):
        self.theta[i] = th
        self._P[i] = tension_map(self.geom, th[1:3])[i]

    def model_T(self, i, a_ref):
        """a_ref (M,3) -> (M,)"""
        a_up = np.atleast_2d(a_ref) + np.array([0, 0, self.geom.g])
        return np.maximum(((self.geom.m_nom + self.theta[i, 0]) * a_up) @ self._P[i], 0.0)

    def update(self, y, a_ref, q):
        """y (n,3)、a_ref (3,)、q (n,3)：同一個（可能延遲的）量測包。"""
        res = np.array([y[i] - self.model_T(i, a_ref[None])[0] * q[i] for i in range(self.n)])
        if not self.init:
            self.r, self.init = res.copy(), True
        else:
            self.r += self.alpha * (res - self.r)

    def force_fn(self, i):
        def fn(t, ts, a_ref, q):
            Tm = self.model_T(i, a_ref)
            dec = np.exp(-(np.asarray(ts) - ts[0]) / self.tau)
            return Tm[:, None] * q[None] + self.r[i][None] * dec[:, None]
        return fn
