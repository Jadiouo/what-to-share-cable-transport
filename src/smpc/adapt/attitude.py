"""P2b-B：payload 姿態感知的 CEN 回歸模型（離線 gap 檢查用）。

姿態只用各機位置與量到的繩方向估：掛點世界位置 pa_i = pq_i + L q_i（繩拉直時），再用 Kabsch 對齊名義掛點 -> (R̂, p̂c)。
不讀 payload 真值。只在 CEN 設定可用（需要全隊的位置、繩方向、IMU）。

V1 ATT（姿態感知）：
  (a) 張力列：用「實際」繩方向 w_i = -q_i 與掛點相對位置 rel_i = pa_i - p̂c 以及 R̂ 取代名義幾何，解
          A = [w^T; ((rel_i - R̂ c) x w_i)^T]，P = pinv(A)[:, :3]，T^model = P (m0 + δm)(a_ref + g e_z)
  (b) 力矩偽量測列（c 的主要資訊來源）：繩對 payload 的力 = -y_i，準靜態力矩平衡
          sum_i (rel_i - R̂ c) x (-y_i) = 0   =>   sum_i rel_i x y_i = (R̂ c) x S，S = sum_i y_i
      即 tau_meas = G c，G = [ (R̂ e1) x S, (R̂ e2) x S ]。雜訊 sigma_tau^2 = sigma_f^2 sum|rel_i|^2 + sigma_pa^2 sum|y_i|^2，
      sigma_pa = L sigma_q / sqrt2（方向雜訊 -> 掛點位置）。由雜訊模型算出，沒有調參。
V2 LOADDEV（每機負載分配偏差）：參數 [δm, d_1..d_N, ζ_1..ζ_N]，T_i = P_i(0)(m0+δm)(a_ref+g e_z) + d_i；
  δc 以 c~ = argmin sum_t |J_c(t) c - d̂|^2（J_c = ∂T/∂c 在名義幾何、沿參考軌跡）從 d̂ 回推，只為了和 dc 門檻比較。
"""
from __future__ import annotations

import numpy as np

from smpc.adapt.estimators import Geometry, PRIOR_STD, ZETA_STD, tension_map, tension_model
from smpc.adapt.rls import InfoRLS


def estimate_pose(pq, q, L, attach_body):
    pa = pq + L * q
    ca, cp = attach_body.mean(0), pa.mean(0)
    Hm = (attach_body - ca).T @ (pa - cp)
    U, _, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cp - R @ ca, pa


def tension_map_actual(w, rel, Rc, c):
    """w (N,3) payload->quad 單位向量；rel (N,3) 掛點相對幾何中心（世界）；Rc: 3x2 = R̂[:, :2]；c (2,)。回傳 P (N,3)。"""
    N = len(w)
    cv = Rc @ np.asarray(c, float)
    A = np.zeros((6, N))
    A[:3] = w.T
    A[3:] = np.cross(rel - cv, w).T
    return np.linalg.pinv(A)[:, :3]


def tension_model_actual(m0, th, a_up, w, rel, Rc, h=1e-3):
    P0 = tension_map_actual(w, rel, Rc, th[1:3])
    f = (m0 + th[0]) * a_up
    T = P0 @ f
    J = np.zeros((len(w), 3))
    J[:, 0] = P0 @ a_up
    for k in range(2):
        c2 = th[1:3].copy(); c2[k] += h
        J[:, 1 + k] = (tension_map_actual(w, rel, Rc, c2) @ f - T) / h
    return T, J


class ReplayCEN:
    """離線 CEN。variant: 'base'（P2a-2 原版）| 'att'（V1）| 'dev'（V2）。update(a_ref,q,f,y,active,pq) 每步呼叫。"""

    def __init__(self, geom: Geometry, L, variant="base", lam=0.9995, sigma_f=0.15, sigma_q=0.005, att_torque=True, prior_scale=1.0):
        self.geom, self.L, self.variant, self.sigma_f, self.sigma_q = geom, L, variant, sigma_f, sigma_q
        self.att_torque = att_torque
        N = self.n = geom.n
        if variant == "dev":
            pr = np.concatenate([[PRIOR_STD[0]], np.full(N, 0.5), np.full(N, ZETA_STD)])
            self.ns = 1 + N
        else:
            pr = np.concatenate([PRIOR_STD, np.full(N, ZETA_STD)])
            self.ns = 3
        pr = pr * prior_scale     # P2d 調參：先驗標準差整體縮放（預設 1 = 不變）
        self.rls = InfoRLS(np.diag(1 / pr**2), np.zeros(len(pr)), lam)
        self.x = np.zeros(len(pr))
        self.a_hist = []

    @property
    def theta(self):
        """(δm, δc_x, δc_y)；dev 變體的 δc 另用 dc_from_dev() 回推。"""
        return self.x[:3] if self.variant != "dev" else np.array([self.x[0], np.nan, np.nan])

    @property
    def zeta(self):
        return self.x[self.ns:]

    def update(self, a_ref, q, f, y, active, pq):
        N, g = self.n, self.geom.g
        a_up = np.asarray(a_ref, float) + np.array([0, 0, g])
        ns, d = self.ns, self.ns + N
        rows, ys, Rinv = [], [], []
        x = self.x
        w_act = active / self.sigma_f**2
        if self.variant == "dev":
            P0 = tension_map(self.geom, np.zeros(2))
            T0 = P0 @ ((self.geom.m_nom + x[0]) * a_up)
            for i in range(N):
                Hi = np.zeros((3, d))
                Hi[:, 0] = q[i] * (P0[i] @ a_up)
                Hi[:, 1 + i] = q[i]
                Hi[:, ns + i] = f[i]
                # 線性：y_i = q_i (P0_i (m0+δm) a_up + d_i) + ζ f_i；重新線性化只對 δm 有（已是線性），故直接用原 y
                rows.append(Hi); ys.append(y[i] - q[i] * (P0[i] @ a_up) * self.geom.m_nom)
            Hc = np.concatenate(rows); yc = np.concatenate(ys)
            Ri = np.diag(np.repeat(w_act, 3))
            self.rls.update(Hc, yc, Ri)
            self.x = self.rls.estimate
            self.a_hist.append(a_up.copy())
            return
        if self.variant == "att":
            R, pc, pa = estimate_pose(pq, q, self.L, self.geom.attach_body)
            w, rel, Rc = -q, pa - pc, R[:, :2]
        for i in range(N):
            if self.variant == "att":
                Tm, J = tension_model_actual(self.geom.m_nom, x[:3], a_up, w, rel, Rc)
                Ti, Ji = Tm[i], J[i]
            else:
                Ti, Ji = tension_model(self.geom, x[:3], a_ref, i)
            Hi = np.zeros((3, d))
            Hi[:, :3] = np.outer(q[i], Ji)
            Hi[:, ns + i] = f[i]
            rows.append(Hi); ys.append(y[i] - (Ti - Ji @ x[:3]) * q[i])
        Hc = np.concatenate(rows); yc = np.concatenate(ys)
        Ri = np.diag(np.repeat(w_act, 3))
        if self.variant == "att" and self.att_torque and active.all():
            yy = y - x[ns:][:, None] * f                      # 去掉本地 ζ̂ f 後的繩力
            S = yy.sum(0)
            tau = np.cross(rel, yy).sum(0)
            G = np.stack([np.cross(R[:, 0], S), np.cross(R[:, 1], S)], 1)   # 3x2
            Ht = np.zeros((3, d)); Ht[:, 1:3] = G
            s_pa = self.L * self.sigma_q / np.sqrt(2)
            s2 = self.sigma_f**2 * (rel**2).sum() + s_pa**2 * (yy**2).sum()
            Hc = np.concatenate([Hc, Ht]); yc = np.concatenate([yc, tau])
            Ri = np.block([[Ri, np.zeros((3 * N, 3))], [np.zeros((3, 3 * N)), np.eye(3) / s2]])
        self.rls.update(Hc, yc, Ri)
        self.x = self.rls.estimate

    def dc_from_dev(self, a_refs):
        """V2：由 d̂ 回推 δc（最小平方，沿參考軌跡的名義 ∂T/∂c）。"""
        d_hat = self.x[1:1 + self.n]
        dm = self.x[0]
        rows = []
        for a_ref in a_refs[::10]:
            a_up = np.asarray(a_ref) + np.array([0, 0, self.geom.g])
            _, J = tension_model(self.geom, np.array([dm, 0, 0]), a_ref)
            rows.append(J[:, 1:3])
        Jm = np.concatenate(rows)
        return np.linalg.lstsq(Jm, np.tile(d_hat, len(rows)), rcond=None)[0]
