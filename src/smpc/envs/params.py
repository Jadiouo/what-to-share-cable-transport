"""1 kg 級通用 quadrotor 參數。

沒有對應特定機型：研究問題不依賴某台真機，所以數值取「1 kg 級 X 型機」的合理量級，
並讓各數字彼此自洽（慣量由質量與臂長推得）。標「假設」者可以在之後的 sweep 中改動。
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuadParams:
    # 假設：1.0 kg。整數方便心算 mg = 9.81 N，測試門檻也好推。
    mass: float = 1.0
    # 假設：標準重力。用 9.81 而非 9.80665，與測試、MPPI 的名義模型共用同一個常數。
    g: float = 9.81
    # 假設：馬達到機體中心距離 0.20 m（對角軸距 400 mm，約 1 kg 級 10 吋槳機型）。
    arm: float = 0.20
    # 假設：四顆馬達（含 ESC）各 0.1 kg 集中在臂端，其餘質量為中央機身，
    # 近似半徑 0.05 m、高 0.06 m 的實心圓柱。
    # 由此推得 Ixx = Iyy ≈ 0.0086、Izz ≈ 0.0168 kg m^2。
    # 中央機身必須算進去：只用點質量時 Izz = 2*Ixx 剛好卡在三角不等式邊界，MuJoCo 會拒絕。
    # 好處：慣量與質量、臂長自洽，改參數時用 derived_inertia() 重算即可。
    motor_mass: float = 0.1
    # 假設：反扭矩係數 km = 反扭矩 / 推力 [m]。10 吋槳典型 0.01~0.02，取 0.016。
    # 控制介面是推力 (N)，所以只需要這個比值，不需要 rpm 模型。
    km: float = 0.016
    # 假設：推力係數 T = kf * w^2 [N/(rad/s)^2]。env 直接吃推力，所以目前不使用；
    # 保留是為了之後若要加馬達轉速/一階延遲時有一致的換算。
    kf: float = 6.0e-6
    # 假設：單馬達推力上限 8 N（推重比 ≈ 3.3，競速以外的一般機型常見 2~4）。
    # 留足餘裕，讓 η=0.6 降效時仍有控制餘量。
    fmax: float = 8.0

    def derived_inertia(self) -> np.ndarray:
        """對角慣量 (Ixx, Iyy, Izz)，由 motor_mass 與 arm 推得。"""
        d2 = (self.arm / np.sqrt(2.0)) ** 2
        m_body = self.mass - 4 * self.motor_mass
        r, h = 0.05, 0.06
        ixx = 4 * self.motor_mass * d2 + m_body * (3 * r ** 2 + h ** 2) / 12
        izz = 4 * self.motor_mass * self.arm ** 2 + 0.5 * m_body * r ** 2
        return np.array([ixx, ixx, izz])

    @property
    def inertia(self) -> np.ndarray:
        return self.derived_inertia()

    @property
    def rotor_xy(self) -> np.ndarray:
        """(4,2) 馬達位置（body frame，x 前 y 左 z 上）。X 型：位於對角線上。

        順序 0:(+,+) 1:(-,+) 2:(-,-) 3:(+,-)，相鄰馬達轉向相反。
        """
        d = self.arm / np.sqrt(2.0)
        return np.array([[d, d], [-d, d], [-d, -d], [d, -d]])

    @property
    def rotor_spin(self) -> np.ndarray:
        """各馬達的反扭矩符號：作用在機身的 z 向反扭矩 = spin * km * 推力。"""
        return np.array([1.0, -1.0, 1.0, -1.0])

    @property
    def allocation(self) -> np.ndarray:
        """4x4 分配矩陣 A：[總推力; tau_x; tau_y; tau_z] = A @ u（u 為四個 rotor 推力）。

        tau = r x F，F = (0,0,T)：tau_x = y*T，tau_y = -x*T。
        放在 params 而非 controller 裡，因為 env 的 MJCF 與 mixer 必須共用同一份幾何，
        否則兩邊符號不一致時控制器會默默失穩。
        """
        xy = self.rotor_xy
        return np.vstack([np.ones(4), xy[:, 1], -xy[:, 0], self.rotor_spin * self.km])
