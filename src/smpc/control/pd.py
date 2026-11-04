"""幾何 PD 位置/姿態控制器 + mixer（Lee et al. 2010 的 SO(3) 幾何控制，去掉積分項）。

流程：位置 PD -> 期望合力 F_des (world) -> 總推力 T = F_des·b3、期望機體 z 軸
b3_d = F_des/|F_des| -> 與期望偏航組成 R_d -> 姿態誤差 -> 力矩 -> mixer -> 四個 rotor 推力。

選幾何而非 Euler 角級聯：不會在大傾角奇異，之後 MPPI/吊掛的大幅機動也能沿用。
刻意不含積分項：保持「純 PD」，馬達降效測試要照實反映無適應時的能力。
"""
from __future__ import annotations

import numpy as np

from smpc.envs.params import QuadParams


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(w,x,y,z) -> 旋轉矩陣 R（body -> world）。"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def vee(S: np.ndarray) -> np.ndarray:
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


class GeometricPD:
    def __init__(self, params: QuadParams, kp=(6.0, 6.0, 10.0), kd=(4.0, 4.0, 6.0),
                 k_att=150.0, k_rate=20.0, max_tilt_deg=45.0):
        """kp/kd：位置/速度增益 (1/s^2, 1/s)，z 較大以更快抵抗推力不足造成的掉高。
        k_att/k_rate：姿態增益，乘上慣量對角（=角加速度增益），自然頻率 ≈ sqrt(150) ≈ 12 rad/s，
        比位置迴路（~2.5 rad/s）快約 5 倍，符合級聯時間尺度分離。
        max_tilt：限制期望傾角，避免大位置誤差要求不可能的姿態而推力過飽和。"""
        self.p = params
        self.kp, self.kd = np.asarray(kp, float), np.asarray(kd, float)
        self.k_att, self.k_rate = k_att, k_rate
        self.max_tilt = np.deg2rad(max_tilt_deg)
        self.J = np.diag(params.inertia)
        self.A_inv = np.linalg.inv(params.allocation)
        self.saturated = False  # 最近一次 mixer 是否碰到飽和（供測試/診斷）

    def wrench(self, state, pos_d, vel_d=None, acc_d=None, yaw_d=0.0):
        """回傳 (總推力 T [N], 力矩 tau [N m, body frame])。"""
        p = self.p
        pos, q, vel, omega = state[0:3], state[3:7], state[7:10], state[10:13]
        vel_d = np.zeros(3) if vel_d is None else vel_d
        acc_d = np.zeros(3) if acc_d is None else acc_d
        a_cmd = self.kp * (pos_d - pos) + self.kd * (vel_d - vel) + acc_d
        F = p.mass * (a_cmd + np.array([0, 0, p.g]))
        # 限制傾角：水平分量不得超過 tan(max_tilt) * 垂直分量。
        Fz = max(F[2], 0.1 * p.mass * p.g)
        Fxy = F[:2]
        lim = np.tan(self.max_tilt) * Fz
        n = np.linalg.norm(Fxy)
        if n > lim:
            Fxy = Fxy * lim / n
        F = np.array([Fxy[0], Fxy[1], Fz])

        R = quat_to_rot(q)
        b3 = R[:, 2]
        T = float(F @ b3)  # 只取沿目前機體 z 軸的分量，才是機體實際推得出的

        b3d = F / np.linalg.norm(F)
        c1 = np.array([np.cos(yaw_d), np.sin(yaw_d), 0.0])
        b2d = np.cross(b3d, c1)
        b2d /= np.linalg.norm(b2d)
        Rd = np.column_stack([np.cross(b2d, b3d), b2d, b3d])

        e_R = 0.5 * vee(Rd.T @ R - R.T @ Rd)
        e_W = omega  # 期望角速度視為 0（不做前饋），實作簡單且對 PD 夠用
        tau = self.J @ (-self.k_att * e_R - self.k_rate * e_W) + np.cross(omega, self.J @ omega)
        return T, tau

    def mix(self, T: float, tau: np.ndarray) -> np.ndarray:
        """(T, tau) -> 四個 rotor 推力，含飽和處理。

        飽和時優先保推力、犧牲力矩：先夾 T 到可行範圍，再二分搜尋把力矩縮放到所有馬達
        都在 [0, fmax]（姿態比高度更容易在力矩被截斷時失穩，但全部不縮放地硬夾會讓
        力矩方向被扭曲；等比縮放至少保住方向）。"""
        fmax = self.p.fmax
        T = float(np.clip(T, 0.0, 4 * fmax))
        w = np.concatenate([[T], tau])
        u = self.A_inv @ w
        self.saturated = bool(u.min() < 0 or u.max() > fmax)
        if self.saturated:
            u_T = self.A_inv @ np.concatenate([[T], np.zeros(3)])  # 純推力部分，均勻 T/4
            lo, hi = 0.0, 1.0
            for _ in range(30):
                s = 0.5 * (lo + hi)
                u_s = u_T + s * (u - u_T)
                if u_s.min() >= 0 and u_s.max() <= fmax:
                    lo = s
                else:
                    hi = s
            u = u_T + lo * (u - u_T)
        return np.clip(u, 0.0, fmax)

    def __call__(self, state, pos_d, vel_d=None, acc_d=None, yaw_d=0.0) -> np.ndarray:
        return self.mix(*self.wrench(state, pos_d, vel_d, acc_d, yaw_d))
