"""payload 參考軌跡（解析，含速度與加速度）。起點一律是 center，從靜止出發。

ref(t) -> (p, v, a)，t 可為純量或 (M,) 陣列，回傳 (M,3)。
加速度量級 ~1–1.5 m/s^2（派工單）：
  p2p     : 5 次多項式（min-jerk），水平 2 m + 升 0.3 m，T=3.2 s，峰值加速度 = 5.77 * 2/3.2^2 ≈ 1.13
  circle  : 半徑 0.8 m，角速度 w=1.2 rad/s（向心加速度 1.15），角速度 2 s 內平滑升起
  fig8    : x=0.8 sin(φ)，y=0.4 sin(2φ)，w=0.9（y 向峰值加速度 1.3），同樣平滑升起
"""
from __future__ import annotations

import numpy as np

Z0 = 1.0


class Traj:
    def __init__(self, name: str, center=(0.0, 0.0, Z0), speed: float = 1.0):
        assert name in ("hover", "p2p", "circle", "fig8")
        self.name, self.c = name, np.asarray(center, float)
        self.T_ramp = 2.0
        if name == "p2p":
            self.delta, self.Tm = np.array([2.0, 0.0, 0.3]), 3.2
        self.speed = float(speed)   # P2b：circle / fig8 的角速度倍率（加速度 ~ speed^2）；p2p 不受影響
        self.R, self.w_c = 0.8, 1.2 * self.speed
        self.A, self.B, self.w_8 = 0.8, 0.4, 0.9 * self.speed

    # φ(t), φ', φ''：角速度平滑升起 ω(t)=wm*(3x^2-2x^3), x=t/Tr，之後定速
    def _phase(self, t, wm):
        Tr = self.T_ramp
        x = np.clip(t / Tr, 0, 1)
        h = 3 * x ** 2 - 2 * x ** 3
        phi = np.where(t < Tr, wm * Tr * (x ** 3 - x ** 4 / 2), wm * (0.5 * Tr + (t - Tr)))
        dphi = wm * h
        ddphi = np.where(t < Tr, wm * (6 * x - 6 * x ** 2) / Tr, 0.0)
        return phi, dphi, ddphi

    def ref(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        M = len(t)
        p, v, a = (np.tile(self.c, (M, 1)), np.zeros((M, 3)), np.zeros((M, 3)))
        if self.name == "p2p":
            s = np.clip(t / self.Tm, 0, 1)
            f = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
            df = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / self.Tm
            ddf = (60 * s - 180 * s ** 2 + 120 * s ** 3) / self.Tm ** 2
            inside = (t < self.Tm)[:, None]
            p = self.c + f[:, None] * self.delta
            v = inside * df[:, None] * self.delta
            a = inside * ddf[:, None] * self.delta
        elif self.name == "circle":
            ph, dp, ddp = self._phase(t, self.w_c)
            R = self.R
            p[:, 0] += R * (np.cos(ph) - 1); p[:, 1] += R * np.sin(ph)
            v[:, 0] = -R * dp * np.sin(ph); v[:, 1] = R * dp * np.cos(ph)
            a[:, 0] = R * (-ddp * np.sin(ph) - dp ** 2 * np.cos(ph))
            a[:, 1] = R * (ddp * np.cos(ph) - dp ** 2 * np.sin(ph))
        elif self.name == "fig8":
            ph, dp, ddp = self._phase(t, self.w_8)
            A, B = self.A, self.B
            p[:, 0] += A * np.sin(ph); p[:, 1] += B * np.sin(2 * ph)
            v[:, 0] = A * dp * np.cos(ph); v[:, 1] = 2 * B * dp * np.cos(2 * ph)
            a[:, 0] = A * (ddp * np.cos(ph) - dp ** 2 * np.sin(ph))
            a[:, 1] = 2 * B * (ddp * np.cos(2 * ph) - 2 * dp ** 2 * np.sin(2 * ph))
        return p, v, a

    def final(self, T):
        """時間 T 的參考位置（p2p 為終點；其餘為該時刻的點）。"""
        return self.ref(T)[0][0]
