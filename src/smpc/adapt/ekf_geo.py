"""EKF-L+GEO（P2d 拆解對照）：本地 EKF-L 的全部資訊，再加上「隊友位置」，其他一概不收。

目的：CON/CEN-ATT 比 EKF-L 多了兩種東西：(1) 隊友的位置／繩向 -> payload 姿態（Kabsch）-> 姿態感知的張力分配；
(2) 隊友的量測（y、f）與估計資訊（共識）。本方法只給 (1)，用來把「多一條幾何資訊」和「估計共享」的收益拆開：
    EKF-L+GEO − EKF-L  = 隊友位置（幾何）的收益
    CON-ATT − EKF-L+GEO = 估計／量測共享的收益（推論：不含幾何）

方法 × 可用資訊（✓ = 該機的估計器會讀；✗ = 不讀）
| 方法        | IMU(y,f) | 自身位姿 | 繩向 | 隊友位置(pq_j,q_j) | 隊友估計 | 隊友量測(y_j,f_j,active_j) |
| DOB         | ✓(自己)  | ✗        | ✗    | ✗                  | ✗        | ✗                          |
| LOC-M       | ✓(自己)  | ✓        | ✓    | ✗                  | ✗        | ✗                          |
| EKF-L       | ✓(自己)  | ✓        | ✓    | ✗                  | ✗        | ✗                          |
| EKF-L+GEO   | ✓(自己)  | ✓        | ✓    | ✓(轉傳 N-1 輪)     | ✗        | ✗                          |
| CEN-ATT     | ✓(全隊)  | ✓        | ✓    | ✓(集中)            | (集中)   | ✓(集中)                    |
| CON-ATT     | ✓(自己)  | ✓        | ✓    | ✓(轉傳 N-1 輪)     | ✓(資訊對 Λ,ξ，共識) | ✓(轉傳 N-1 輪)  |
註：(a) 「位置」＝隊友機位 pq_j 與其繩向 q_j（合成掛點 pa_j=pq_j+L q_j）；姿態感知的張力分配需要全隊的繩向 w_j=-q_j，所以 q_j 也算在封包裡
    （共 6 個數／來源）。這比純機位多一點，CON-ATT 的封包也含它。(b) DOB 不用位姿與繩向（只用 y 的低通）。
    (c) 所有方法都不讀 payload 真值。

設計：
  狀態 x_i = [δm, ζ_i, b_i(3), (δc_x, δc_y)]（use_dc=1 時多兩維；prior = estimators.PRIOR_STD[1:]）。過程模型與量測噪聲同 EKF-L。
  量測模型：y_i = T_i(δm, δc; 姿態) q_i + ζ_i f_i + m_q b_i + 雜訊，其中 T_i 取自 attitude.tension_model_actual：
  用本機轉傳得到的全隊 (pq_j,q_j) 做 Kabsch 得 (R̂, p̂c)，以實際繩向 w=-q 與掛點相對位置 rel 解力與力矩平衡的張力分配（偽逆），
  取代 EKF-L 用的名義幾何張力分配。δc 對 T 的影響以數值微分線性化（EKF 的雅可比），δm 精確線性。
  沒有力矩偽量測（那需要隊友的 y_j，違反資訊條件）。δc 只能從自己的 y_i 與姿態感知的 T_i 估，可觀測性弱，靠 prior 約束。
  轉傳：每步 relay_rounds（預設 N-1）輪，每輪鄰居（path 圖）把手上的表給我，每個來源取時間戳較新的一筆；掉包不實作（p_drop=0）。
  任何來源缺或過期（> max_age 步）-> 該機該步退回名義幾何（同 EKF-L，δc 不動）。slack（active=False）只做預測不更新。
"""
from __future__ import annotations

import numpy as np

from smpc.adapt.attitude import estimate_pose, tension_model_actual
from smpc.adapt.estimators import PRIOR_STD, Geometry, tension_model

DEFAULTS = dict(q_b=5e-4, prior_dm=0.5, prior_zeta=0.3, q_dm=1e-4, q_zeta=1e-4, r_scale=10.0, use_dc=1, q_dc=1e-5)
PRIOR_B = 0.2


class EKFLGeo:
    """update(a_ref, q, f, y, active, pq) 每步呼叫（陣列含全隊；每機只讀自己那一列的 y、f、active，其他機只經 table 的 (pq,q) 進來）。"""

    def __init__(self, geom: Geometry, L: float, m_q: float, sigma_f: float, hp: dict | None = None, max_age=5, relay_rounds=None):
        h = {**DEFAULTS, **{k: v for k, v in (hp or {}).items() if k in DEFAULTS}}
        self.geom, self.L, self.n, self.m_q, self.max_age = geom, float(L), geom.n, float(m_q), max_age
        N = self.n
        self.relay_rounds = (N - 1) if relay_rounds is None else int(relay_rounds)
        self.use_dc = bool(h["use_dc"])
        self.d = 7 if self.use_dc else 5
        self.R = (h["r_scale"] * sigma_f) ** 2
        qd = [h["q_dm"] ** 2, h["q_zeta"] ** 2] + [h["q_b"] ** 2] * 3 + ([h["q_dc"] ** 2] * 2 if self.use_dc else [])
        self.Q = np.diag(qd)
        p0 = [h["prior_dm"] ** 2, h["prior_zeta"] ** 2] + [PRIOR_B ** 2] * 3 + (list(PRIOR_STD[1:3] ** 2) if self.use_dc else [])
        self.x = np.zeros((N, self.d))
        self.P = np.broadcast_to(np.diag(p0), (N, self.d, self.d)).copy()
        self.theta = np.zeros((N, 3))
        self.zeta = np.zeros(N)
        self.bias = np.zeros((N, 3))
        adj = np.eye(N, k=1) + np.eye(N, k=-1)
        self.adj = adj.astype(bool)
        self.k = 0
        self.stamp = np.full((N, N), -10**9, dtype=np.int64)       # stamp[i, o]：機 i 手上機 o 封包的時間戳
        self.data = np.zeros((N, N, 6))                            # 封包只有 (pq, q)：沒有 y、f、active、估計
        self.pose_ok_frac = []

    def _relay(self):
        for _ in range(self.relay_rounds):
            st, da = self.stamp.copy(), self.data.copy()
            for i in range(self.n):
                for j in np.flatnonzero(self.adj[i]):
                    new = self.stamp[j] > st[i]
                    st[i] = np.where(new, self.stamp[j], st[i])
                    da[i][new] = self.data[j][new]
            self.stamp, self.data = st, da

    def update(self, a_ref, q, f, y, active=None, pq=None):
        N, g = self.n, self.geom.g
        active = np.ones(N, bool) if active is None else np.asarray(active, bool)
        self.k += 1
        for i in range(N):                                         # 每機只放自己的 (pq_i, q_i)
            self.stamp[i, i] = self.k
            self.data[i, i] = np.concatenate([pq[i], q[i]])
        self._relay()
        a_up = np.asarray(a_ref, float) + np.array([0, 0, g])
        I3 = np.eye(3)
        oks = []
        for i in range(N):
            self.P[i] = self.P[i] + self.Q
            ok = bool(np.all(self.k - self.stamp[i] <= self.max_age))
            oks.append(ok)
            if not active[i]:
                continue
            x = self.x[i]
            th = np.array([x[0], x[5], x[6]]) if self.use_dc else np.array([x[0], 0.0, 0.0])
            if ok:
                pqs, qs = self.data[i, :, 0:3], self.data[i, :, 3:6]
                Rm, pc, pa = estimate_pose(pqs, qs, self.L, self.geom.attach_body)
                Tm, J = tension_model_actual(self.geom.m_nom, th, a_up, -qs, pa - pc, Rm[:, :2])
                Ti, Ji = Tm[i], J[i]
            else:
                Ti, Ji = tension_model(self.geom, th, a_ref, i)
            H = np.zeros((3, self.d))
            H[:, 0] = q[i] * Ji[0]
            H[:, 1] = f[i]
            H[:, 2:5] = self.m_q * I3
            if self.use_dc:
                H[:, 5] = q[i] * Ji[1]
                H[:, 6] = q[i] * Ji[2]
            innov = np.asarray(y[i], float) - (Ti * q[i] + x[1] * f[i] + self.m_q * x[2:5])
            S = H @ self.P[i] @ H.T + self.R * I3
            K = np.linalg.solve(S, H @ self.P[i]).T
            self.x[i] = x + K @ innov
            A = np.eye(self.d) - K @ H
            self.P[i] = A @ self.P[i] @ A.T + self.R * (K @ K.T)
            self.P[i] = 0.5 * (self.P[i] + self.P[i].T)
        self.pose_ok_frac.append(float(np.mean(oks)))
        self.theta[:, 0] = self.x[:, 0]
        if self.use_dc:
            self.theta[:, 1:3] = self.x[:, 5:7]
        self.zeta = self.x[:, 1].copy()
        self.bias = self.x[:, 2:5].copy()
