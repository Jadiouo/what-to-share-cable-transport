"""吊掛懸停：既有幾何 PD + 已知 payload 的靜力前饋（每機多出 T_i * q_i 方向的推力需求）。

前饋只用「已知的 payload 質量／重心」與名義幾何算出的張力，不讀模擬器的張力真值。
"""
from __future__ import annotations

import numpy as np

from smpc.control.pd import GeometricPD
from smpc.envs.multi_payload import MultiPayloadEnv


class SuspendedHover:
    def __init__(self, env: MultiPayloadEnv, cone_deg: float = 30.0, feedforward: bool = True):
        self.env, self.cone = env, cone_deg
        self.pd = [GeometricPD(env.p) for _ in range(env.n)]
        self.set_known_payload(env.payload_mass, env.payload_com, feedforward)

    def set_known_payload(self, mass, com, feedforward=True):
        """前饋所依據的 payload（可與真值不同，以模擬「估計有偏差」）。"""
        e = self.env
        save = (e.payload_mass, e.payload_com)
        e.payload_mass, e.payload_com = float(mass), np.asarray(com, float)
        T = e.static_tensions(self.cone)
        e.payload_mass, e.payload_com = save
        # 繩對 quad 的力 = T * q（q: quad->payload 單位向量）；quad 需用推力抵銷 => acc_d = -T q / m
        self.acc_ff = -(T[:, None] * e.rope_unit_nominal(self.cone)) / e.p.mass if feedforward \
            else np.zeros((e.n, 3))

    def __call__(self, quad_states, targets) -> np.ndarray:
        return np.stack([self.pd[i](quad_states[i], targets[i], acc_d=self.acc_ff[i])
                         for i in range(self.env.n)])
