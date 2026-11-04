"""P2b 強的不通訊基線（每機只用自己的量測，沒有任何跨機資訊）。

量測沿用 P2a-2：力殘差 y_i = m a_imu,i - f_i = T_i q_i + zeta_i f_i + 雜訊（沒有降效時 zeta=0）。

  LocalDm       IND-δm / IND-tight-δc0：每機只估 [δm, ζ_i]（δc 固定為 0），資訊形式 RLS + 沿參考軌跡重新線性化。
                prior_dm = 0.5（IND-δm）或 0.25（tight）。
  DOBHold       DOB-hold：y_i 經一階低通直接當作自己的繩力向量，預測時域內保持常數。
  LoadShare     LOADSHARE（GPAC 風格）：s_i = LPF( (y_i . q_i) / (m_nom |a_ref + g e_z|) )，預測 T_i(t_k) = s_i m_nom |a_ref(t_k) + g e_z|。
  LoadShareNom  LOADSHARE-N（比規格更強的變體，另外報告）：s_i = LPF(T_i / T_i^nom(t))，預測 T_i(t_k) = s_i T_i^nom(t_k)，
                T^nom 是名義編隊幾何、θ=0 的張力映射（機上可得：編隊幾何與參考軌跡都是事前約定）。
"""
from __future__ import annotations

import numpy as np

from smpc.adapt.estimators import Geometry, tension_model
from smpc.adapt.rls import InfoRLS


class LocalDm:
    """每機獨立估 [δm, ζ_i]；θ_i = (δm_i, 0, 0)。介面同 SharedEstimator：update(a_ref, q, f, y, active)。"""

    def __init__(self, geom: Geometry, prior_dm=0.5, zeta_std=0.3, lam=0.9995, sigma_f=0.15, prior_scale=1.0):
        prior_dm, zeta_std = prior_dm * prior_scale, zeta_std * prior_scale   # P2d 調參：先驗整體縮放
        self.geom, self.n, self.sigma_f = geom, geom.n, sigma_f
        N = self.n
        P0 = np.diag([1 / prior_dm**2, 1 / zeta_std**2])
        self.rls = InfoRLS(np.broadcast_to(P0, (N, 2, 2)).copy(), np.zeros((N, 2)), lam)
        self.theta = np.zeros((N, 3))
        self.zeta = np.zeros(N)

    def update(self, a_ref, q, f, y, active=None):
        N = self.n
        active = np.ones(N, bool) if active is None else np.asarray(active, bool)
        H = np.zeros((N, 3, 2))
        yp = np.array(y, float)
        for i in range(N):
            th = np.array([self.theta[i, 0], 0.0, 0.0])
            Ti, Ji = tension_model(self.geom, th, a_ref, i)
            H[i, :, 0] = q[i] * Ji[0]
            H[i, :, 1] = f[i]
            yp[i] = y[i] - (Ti - Ji[0] * th[0]) * q[i]
        Rinv = (active / self.sigma_f**2)[:, None, None] * np.eye(3)[None]
        self.rls.update(H, yp, Rinv)
        est = self.rls.estimate
        self.theta[:, 0] = est[:, 0]
        self.zeta = est[:, 1].copy()


class DOBHold:
    """y_i 的一階低通 -> 自己的繩力向量估計（機 -> payload 方向的力）。fc 單位 Hz。"""

    def __init__(self, n, fc, dt=0.01):
        self.alpha = 1 - np.exp(-2 * np.pi * fc * dt)
        self.F = np.zeros((n, 3))
        self.init = False

    def update(self, y, active=None):
        if not self.init:
            self.F = np.array(y, float)
            self.init = True
        else:
            self.F += self.alpha * (np.asarray(y, float) - self.F)

    def force_fn(self, i):
        def fn(t, ts, a_ref, q):
            return np.tile(self.F[i], (len(ts), 1))
        return fn


class LoadShare:
    def __init__(self, n, fc, dt=0.01, m_nom=1.0, g=9.81, geom: Geometry | None = None, nominal=False):
        self.alpha = 1 - np.exp(-2 * np.pi * fc * dt)
        self.n, self.m, self.g, self.geom, self.nominal = n, m_nom, g, geom, nominal
        self.s = np.full(n, np.nan)
        # LSN 的防護（事先指定的物理理由，不是用結果調的）：名義張力 < 1 N 時不更新分配比，比值夾在 [0, 3]。
        # 第一版沒有這個防護，LSN 在 fig8 x1.5 的調參 seeds 上因 T/T_nom 在 T_nom 小時爆掉而發散（傾角 > 100 度），
        # 依 pilot 調參結果（內部紀錄，未公開）。LS（|a+g| 版）的分母 >= 9.8 N，不需要。
        self.d_min = 1.0 if nominal else 0.3
        self._P0 = None
        self.s_max = 3.0
        self.ez = np.array([0, 0, g])

    def demand(self, i, a_ref):
        """a_ref (M,3) -> 需求 (M,)。"""
        a_up = np.atleast_2d(a_ref) + self.ez
        if not self.nominal:
            return self.m * np.linalg.norm(a_up, axis=1)
        if self._P0 is None:
            from smpc.adapt.estimators import tension_map
            self._P0 = tension_map(self.geom, np.zeros(2))
        return (self.geom.m_nom * (a_up @ self._P0[i]))

    def update(self, a_ref, q, y, active=None):
        for i in range(self.n):
            if active is not None and not active[i]:
                continue
            D = self.demand(i, a_ref)[0]
            if D < self.d_min:   # 名義需求太小（只會發生在 nominal 變體）：比值會放大雜訊，不更新
                continue
            raw = float(np.clip(float(y[i] @ q[i]) / D, 0.0, self.s_max))
            self.s[i] = raw if np.isnan(self.s[i]) else self.s[i] + self.alpha * (raw - self.s[i])

    def force_fn(self, i):
        def fn(t, ts, a_ref, q):
            s = 0.0 if np.isnan(self.s[i]) else self.s[i]
            T = s * self.demand(i, a_ref)
            return T[:, None] * q[None]
        return fn
