"""EKF-L（P2d，本地）：每機一個 EKF，狀態 x_i = [δm, ζ_i, b_i(3)]，b_i 是 IMU 偏差（m/s^2）。

量測模型與 LocalDm 一致，只多一個偏差項（IMU 偏差進入 a_imu，所以進入 y 時乘 m_q）：
    y_i = m_q a_imu,i - f_i = T_i(δm) q_i + ζ_i f_i + m_q b_i + 雜訊
T_i(δm) = (T_i^model(θ=(δm,0,0)))，δc 固定為 0（同 LocalDm）。T 對 δm 是線性的（(m0+δm) a_up），所以整個量測對 x 線性，
EKF 的「線性化」是精確的，退化成帶隨機漫步過程噪聲的卡爾曼濾波。
過程模型：隨機漫步 x_{k+1} = x_k + w，Q = diag(q_dm^2, q_zeta^2, q_b^2 I)（每控制步）。不用遺忘因子。
不使用 payload 真值，也不使用其他機的資訊；slack 的步（active=False）只做預測、不更新。

可調超參數（hp，P2d 規格：偏差隨機漫步 Q、ζ 與 δm 的 prior／過程噪聲、量測 R）：
    q_b（偏差隨機漫步，m/s^2 每步）、prior_dm、prior_zeta、q_dm、q_zeta、r_scale（R = (r_scale * sigma_f)^2 I）。
固定（不調）：偏差 prior 標準差 prior_b = 0.2 m/s^2（約 harsh 的穩態偏差 σ，對 moderate 偏保守）。

可觀測性（推論，未證明；P 的事後標準差見 smoke 報告）：
  - 懸停（a_ref ≈ 0）：y 是常數向量。未知數 δm、ζ、b(3) 共 5 個，量測 3 個常數方程式，T q 與 ζ f 與 m_q b 全混在一起：
    垂直分量上 δm·J q_z、ζ f_z、m_q b_z 三者共線，不可分；水平分量上 b_x,b_y 與 ζ f_h、T q_h 也只靠 q、f 的微小傾斜區分。
  - 有機動時（circle、fig8）：a_ref 旋轉使 δm J(t) q(t) 與 ζ f(t) 隨時間變化，而 b 在世界座標系是（近似）常數，
    所以水平偏差可以由「時變部分 vs 常數部分」分開；垂直偏差與 ζ、δm 的垂直分量仍近乎共線（f_z、T q_z 幾乎同方向且變化小）。
  - 實作上偏差在模擬器是 Gauss-Markov（tau 20 s），EKF 用隨機漫步近似：模型失配，q_b 要夠大才跟得上。
"""
from __future__ import annotations

import numpy as np

from smpc.adapt.estimators import Geometry, tension_model

DEFAULTS = dict(q_b=2e-3, prior_dm=0.5, prior_zeta=0.3, q_dm=1e-4, q_zeta=1e-4, r_scale=1.0)
PRIOR_B = 0.2


class EKFL:
    """update(a_ref, q, f, y, active) 每步呼叫；theta (N,3)=(δm_i,0,0)，zeta (N,)，bias (N,3)。"""

    def __init__(self, geom: Geometry, m_q: float, sigma_f: float, hp: dict | None = None):
        h = {**DEFAULTS, **(hp or {})}
        self.geom, self.n, self.m_q = geom, geom.n, float(m_q)
        self.R = (h["r_scale"] * sigma_f) ** 2
        self.Q = np.diag([h["q_dm"] ** 2, h["q_zeta"] ** 2] + [h["q_b"] ** 2] * 3)
        N = self.n
        self.x = np.zeros((N, 5))
        P0 = np.diag([h["prior_dm"] ** 2, h["prior_zeta"] ** 2] + [PRIOR_B ** 2] * 3)
        self.P = np.broadcast_to(P0, (N, 5, 5)).copy()
        self.theta = np.zeros((N, 3))
        self.zeta = np.zeros(N)
        self.bias = np.zeros((N, 3))

    def update(self, a_ref, q, f, y, active=None):
        N = self.n
        active = np.ones(N, bool) if active is None else np.asarray(active, bool)
        I3 = np.eye(3)
        for i in range(N):
            self.P[i] = self.P[i] + self.Q
            if not active[i]:
                continue
            x = self.x[i]
            Ti, Ji = tension_model(self.geom, np.array([x[0], 0.0, 0.0]), a_ref, i)
            T0 = Ti - Ji[0] * x[0]                          # δm=0 時的張力（T 對 δm 線性）
            H = np.zeros((3, 5))
            H[:, 0] = q[i] * Ji[0]
            H[:, 1] = f[i]
            H[:, 2:] = self.m_q * I3
            innov = np.asarray(y[i], float) - (T0 * q[i] + H @ x)
            S = H @ self.P[i] @ H.T + self.R * I3
            K = np.linalg.solve(S, H @ self.P[i]).T          # P H^T S^-1（P、S 對稱）
            self.x[i] = x + K @ innov
            A = np.eye(5) - K @ H
            self.P[i] = A @ self.P[i] @ A.T + self.R * (K @ K.T)   # Joseph 形式
            self.P[i] = 0.5 * (self.P[i] + self.P[i].T)
        self.theta[:, 0] = self.x[:, 0]
        self.zeta = self.x[:, 1].copy()
        self.bias = self.x[:, 2:].copy()
