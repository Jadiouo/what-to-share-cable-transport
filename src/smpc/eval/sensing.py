"""P2c 感測 profile（規格：docs/notes/p2c_prespec.md；數值是假設值，不是實測）。

量測模擬層（只在 episode 執行器裡用）：
  IMU 比力  a_imu = a_spec(真值有限差分) + 偏差 b(t) + 白雜訊
  偏差 b    每機每軸的 Gauss-Markov 隨機漫步  b' = -b/tau_b + sigma_w * w，穩態標準差 = bias_std（規格「穩態約 0.1 m/s^2」）。
            tau_b = 20 s（假設；規格沒給），初值取穩態分佈。偏差加在世界座標系（忽略機體傾角造成的旋轉，簡化）。
  延遲      整包量測（y、f、q、a_ref、active、pq）延遲 delay 個控制步後才交給估計器／DOB。
            f（自己送出的指令）與 a_ref 都是已知量，所以對齊到量測取樣時刻（= 整包延遲）；沒有對齊的版本沒有跑。
  推力係數  每機一個固定係數 k_i ~ U(1-kerr, 1+kerr)：真實推力 = 指令 * k_i，估計器與 DOB 以為 = 指令 * 1。
  機位雜訊  估計器看到的機位 = 真值 + N(0, pos_std^2)（只在 moderate / harsh）；控制器自己的狀態不加雜訊（沿用 P2b）。
  繩方向    角度雜訊 sigma_q（clean 同 P2b）。
clean 與 P2b 完全相同：RNG 流（sens）的抽樣順序不變，新增的雜訊另用 seed 衍生的獨立流，clean 不抽。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Profile:
    name: str
    sigma_a: float      # IMU 白雜訊 (m/s^2)
    bias_std: float     # 偏差穩態標準差 (m/s^2)
    delay: int          # 控制步（0.01 s）
    kerr: float         # 推力係數誤差範圍 (±)
    sigma_q: float      # 繩方向雜訊 (rad)
    pos_std: float      # 機位量測雜訊 (m)


PROFILES = {
    "clean": Profile("clean", 0.15, 0.0, 0, 0.0, 0.005, 0.0),
    "moderate": Profile("moderate", 0.6, 0.1, 2, 0.05, 0.01, 0.01),
    "harsh": Profile("harsh", 1.5, 0.2, 3, 0.10, 0.02, 0.01),
}
TAU_B = 20.0


def scaled_profile(s: float) -> Profile:
    """P2d 感測強度掃描（規格 docs/notes/p2d_prespec.md「感測強度掃描」）：比例 s，s=0 為 clean，s=1 為 moderate。
    IMU σ、偏差、推力係數誤差、繩方向雜訊從 clean 到 moderate 線性外插：v(s) = v_clean + s (v_moderate - v_clean)（s=2.5 約等於 harsh）。
    延遲：s<0.5 為 0 步，0.5<=s<1.75 為 2 步，s>=1.75 為 3 步。
    機位雜訊 pos_std 規格沒列為隨 s 放大：取 0.01 * min(s, 1)（s=0 為 0 以維持 clean 逐位相同，s>=1 為 0.01 同 moderate／harsh）。這是實作選擇。
    s=0 與 s=1 直接回傳 clean／moderate 物件，保證逐位相同（不經浮點外插）。"""
    if s < 0:
        raise ValueError(s)
    if s == 0:
        return PROFILES["clean"]
    if s == 1:
        return PROFILES["moderate"]
    c, m = PROFILES["clean"], PROFILES["moderate"]
    lin = lambda a, b: a + s * (b - a)
    delay = 0 if s < 0.5 else (2 if s < 1.75 else 3)
    return Profile(f"s={s:g}", lin(c.sigma_a, m.sigma_a), lin(c.bias_std, m.bias_std), delay, lin(c.kerr, m.kerr),
                   lin(c.sigma_q, m.sigma_q), m.pos_std * min(s, 1.0))


def get_profile(name: str) -> Profile:
    """名稱（clean／moderate／harsh）或比例字串 "s=1.75"。"""
    if name in PROFILES:
        return PROFILES[name]
    if isinstance(name, str) and name.startswith("s="):
        return scaled_profile(float(name[2:]))
    raise KeyError(name)


class SensorModel:
    def __init__(self, prof: Profile, seed: int, n: int, dt: float):
        self.prof, self.n, self.dt = prof, n, dt
        self.rng = np.random.default_rng([seed, 4])
        rk = np.random.default_rng([seed, 5])
        self.k_true = np.ones(n) if prof.kerr == 0 else rk.uniform(1 - prof.kerr, 1 + prof.kerr, n)
        self.bias = self.rng.normal(0, prof.bias_std, (n, 3)) if prof.bias_std > 0 else np.zeros((n, 3))
        self._sw = prof.bias_std * np.sqrt(2.0 / TAU_B)
        self.buf = deque()

    def step_bias(self):
        if self.prof.bias_std > 0:
            self.bias = self.bias * (1 - self.dt / TAU_B) + self._sw * np.sqrt(self.dt) * self.rng.standard_normal((self.n, 3))
        return self.bias

    def pos_meas(self, pq):
        if self.prof.pos_std == 0:
            return pq
        return pq + self.prof.pos_std * self.rng.standard_normal(pq.shape)

    def delay_bundle(self, bundle):
        """送入這一步的量測包，回傳估計器此刻能用的包（落後 delay 步；開頭不足時用最早的一包）。"""
        self.buf.append(bundle)
        while len(self.buf) > self.prof.delay + 1:
            self.buf.popleft()
        return self.buf[0]
