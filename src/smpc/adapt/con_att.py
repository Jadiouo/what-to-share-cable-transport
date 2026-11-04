"""CON-ATT（P2d，分散式分享）：把 CEN-ATT（attitude.ReplayCEN variant='att'）的估計在 path 圖（1-2-3）上分散化。

共識方式：HCMCI（Battistelli–Chisci 的 hybrid consensus on measurements and information）。
  選它的理由：(1) PLAN R4 已指定 HCMCI；(2) 舊的 estimators.CON 是「非遞迴資訊 consensus」（本地 RLS 不吃融合結果，融合量
  用 Λ_f = N·mean − (N−1)Λ0 重組），PLAN 已註記它和 HCMCI 不一致，且融合結果不回灌所以沒有真正的遞迴狀態；(3) HCMCI 是遞迴的：
  每機維護完整團隊狀態 x = [δm, δc_x, δc_y, ζ_1..ζ_N] 的資訊對 (Λ_i, ξ_i)，融合結果就是下一步的先驗，所以
  理想通訊且迭代次數 → ∞ 時與 CEN 完全一致（tests/test_con_att.py 驗證 < 1%）。
  每步演算法（機 i）：
    1. 預測（遺忘，同 CEN 的 InfoRLS）：Λ⁻_i = λΛ_i + (1−λ)Λ0，ξ⁻_i 同。
    2. 本地新增資訊 (U_i, u_i)：只用自己的量測列（見下）。
    3. 做 `iters`（預設 2）次共識，權重 Metropolis W：
         先驗對 (Λ⁻, ξ⁻) ← W 混合（CI 部分：凸組合，不重複計算舊資訊）
         新增資訊對 (U, u) ← W 混合（CM 部分）
    4. 後驗 Λ_i = [W^iters Λ⁻]_i + N·[W^iters U]_i，ξ_i 同；x̂_i = Λ_i⁻¹ ξ_i。
  W 雙隨機 ⇒ 理想通訊下 Σ_i 後驗資訊 = Σ_i 先驗 + Σ_j U_j（總資訊量與 CEN 相同），iters → ∞ 時每機都等於 CEN。
  每則訊息 = 兩個 (Λ,ξ) 對（各 21+6 個獨立數）＋ 一個量測封包（13 個數）。

CEN-ATT 的兩個跨機耦合，以及怎麼只靠鄰居轉傳取得：
  (a) 姿態：Kabsch 需要全隊的掛點位置 pa_j = pq_j + L q_j；(b) 力矩偽量測需要全隊的 y_j、f_j、pq_j、q_j、active_j。
  做法：每機維護一張「已知表」table[i][o]（最近一次收到的機 o 的量測封包 (pq, q, y, f, active) 與時間戳）。
  每步在共識之前先做 `relay_rounds`（預設 = path 圖直徑 N-1；N=3 時 =2）輪轉傳：每輪鄰居把自己的表給我，我對每個來源 o 採用時間戳較新的那筆。
  理想通訊下 N-1 輪後每機都有全隊當步封包（與 CEN 同資料，無近似）。
  近似假設（要報告）：
    - 量測封包的轉傳是額外的 2 輪通訊（13 個數 × 來源數），不算在「每步 2 次迭代」的共識裡；訊息大小是 O(N)，N=3 時很小。
    - 掉包時用「上一筆收到的值」：表裡的舊封包照用，但超過 max_age（預設 5 步 = 50 ms）就視為沒有；任何一個來源缺或過期，
      該機該步退回「名義幾何張力列」（estimators.tension_model，不需要別機資料），並略過力矩偽量測。
    - 力矩偽量測每機都算一份，資訊權重乘 1/N，所以共識後總共算一份（不重複計算）；各機的 ζ̂_j 用自己的估計去扣 y_j。
  共識的掉包（p_drop）：每條有向訊息（j→i）每次迭代（含轉傳輪）獨立以 p_drop 丟棄。選「略過該鄰居」：丟掉的權重併回自己
  （該列仍是隨機矩陣，不用舊值，所以先驗與新增資訊都不會混進過期的資料）。代價（推論）：W 不再雙隨機，N·[·] 的縮放在高掉包率下
  會把自己的新增資訊算成最多 N 倍（過度自信，估計仍大致不偏）；p_drop=1 時等於一個「資訊 ×N」的本地估計器。
  （沒選「用上一筆值」做資訊共識，因為重用舊的新增資訊等於把同一筆量測算兩次。）
"""
from __future__ import annotations

import numpy as np

from smpc.adapt.attitude import estimate_pose, tension_model_actual
from smpc.adapt.estimators import PRIOR_STD, ZETA_STD, Geometry, tension_model
from smpc.adapt.rls import metropolis_weights


class ConATT:
    """update(a_ref, q, f, y, active, pq) 每步呼叫（陣列含全隊，但每機只讀自己那一列；其他機資料只經 table 轉傳取得）。"""

    def __init__(self, geom: Geometry, L, lam=0.9995, sigma_f=0.15, sigma_q=0.005, iters=2, p_drop=0.0, seed=0,
                 max_age=5, relay_rounds=None, prior_scale=1.0):
        self.geom, self.L, self.sigma_f, self.sigma_q, self.lam = geom, L, sigma_f, sigma_q, lam
        self.iters, self.p_drop, self.max_age = iters, p_drop, max_age
        N = self.n = geom.n
        # 轉傳輪數 = path 圖直徑 N-1（N=3 時為 2，與舊預設逐位相同）；可由 hp 指定（relay_rounds=0 用於局部性測試）
        self.relay_rounds = (N - 1) if relay_rounds is None else int(relay_rounds)
        self.d = 3 + N
        pr = np.concatenate([PRIOR_STD, np.full(N, ZETA_STD)]) * prior_scale
        self.L0 = np.diag(1 / pr**2)
        self.Lam = np.broadcast_to(self.L0, (N, self.d, self.d)).copy()
        self.xi = np.zeros((N, self.d))
        self.x = np.zeros((N, self.d))
        adj = np.eye(N, k=1) + np.eye(N, k=-1)
        self.adj = adj.astype(bool)
        self.W = metropolis_weights(adj)
        self.rng = np.random.default_rng([int(seed), 6])
        self.k = 0
        self.stamp = np.full((N, N), -10**9, dtype=np.int64)      # stamp[i, o]：機 i 手上機 o 封包的時間戳
        self.data = np.zeros((N, N, 13))
        self.pose_ok_frac = []                                    # 診斷：各步各機是否用了姿態感知列
        self.n_pose = 0

    # ---- 通訊 -----------------------------------------------------------
    def _recv(self):
        """rec[i, j] = 機 i 這次收到機 j 的訊息。"""
        if self.p_drop <= 0:
            return self.adj
        return self.adj & (self.rng.random((self.n, self.n)) >= self.p_drop)

    def _mix_matrix(self, rec):
        Wk = np.where(rec, self.W, 0.0)
        np.fill_diagonal(Wk, 0.0)
        np.fill_diagonal(Wk, 1.0 - Wk.sum(1))
        return Wk

    def _relay(self):
        for _ in range(self.relay_rounds):
            rec = self._recv()
            st, da = self.stamp.copy(), self.data.copy()
            for i in range(self.n):
                for j in np.flatnonzero(rec[i]):
                    new = self.stamp[j] > st[i]
                    st[i] = np.where(new, self.stamp[j], st[i])
                    da[i][new] = self.data[j][new]
            self.stamp, self.data = st, da

    # ---- 本地新增資訊 ---------------------------------------------------
    def _local_info(self, i, a_ref, x):
        N, d, g = self.n, self.d, self.geom.g
        age = self.k - self.stamp[i]
        ok = bool(np.all(age <= self.max_age))
        pq, q, y, f, act = (self.data[i, :, 0:3], self.data[i, :, 3:6], self.data[i, :, 6:9], self.data[i, :, 9:12],
                            self.data[i, :, 12] > 0.5)
        a_up = np.asarray(a_ref, float) + np.array([0, 0, g])
        if ok:
            R, pc, pa = estimate_pose(pq, q, self.L, self.geom.attach_body)
            w, rel, Rc = -q, pa - pc, R[:, :2]
            Tm, J = tension_model_actual(self.geom.m_nom, x[:3], a_up, w, rel, Rc)
            Ti, Ji = Tm[i], J[i]
        else:
            Ti, Ji = tension_model(self.geom, x[:3], a_ref, i)
        Hi = np.zeros((3, d))
        Hi[:, :3] = np.outer(q[i], Ji)
        Hi[:, 3 + i] = f[i]
        yi = y[i] - (Ti - Ji @ x[:3]) * q[i]
        wi = float(act[i]) / self.sigma_f**2
        U = wi * Hi.T @ Hi
        u = wi * Hi.T @ yi
        if ok and act.all():
            yy = y - x[3:][:, None] * f
            S = yy.sum(0)
            tau = np.cross(rel, yy).sum(0)
            G = np.stack([np.cross(R[:, 0], S), np.cross(R[:, 1], S)], 1)
            Ht = np.zeros((3, d))
            Ht[:, 1:3] = G
            s_pa = self.L * self.sigma_q / np.sqrt(2)
            s2 = self.sigma_f**2 * (rel**2).sum() + s_pa**2 * (yy**2).sum()
            wt = 1.0 / (N * s2)                                # 每機都算一份，乘 1/N，共識後總共一份
            U = U + wt * Ht.T @ Ht
            u = u + wt * Ht.T @ tau
        return U, u, ok

    # ---- 主更新 ---------------------------------------------------------
    def update(self, a_ref, q, f, y, active, pq):
        N = self.n
        self.k += 1
        for i in range(N):                                     # 自己的封包（只有自己這列）
            self.stamp[i, i] = self.k
            self.data[i, i] = np.concatenate([pq[i], q[i], y[i], f[i], [float(active[i])]])
        self._relay()
        a = self.lam
        Lp = a * self.Lam + (1 - a) * self.L0
        xp = a * self.xi
        U = np.zeros_like(Lp)
        u = np.zeros_like(xp)
        oks = []
        for i in range(N):
            U[i], u[i], ok = self._local_info(i, a_ref, self.x[i])
            oks.append(ok)
        self.pose_ok_frac.append(float(np.mean(oks)))
        for _ in range(self.iters):
            Wk = self._mix_matrix(self._recv())
            Lp, xp = np.einsum("ij,jab->iab", Wk, Lp), Wk @ xp
            U, u = np.einsum("ij,jab->iab", Wk, U), Wk @ u
        self.Lam = Lp + N * U
        self.xi = xp + N * u
        self.x = np.linalg.solve(self.Lam, self.xi[..., None])[..., 0]

    @property
    def theta(self):
        return self.x[:, :3].copy()

    @property
    def zeta(self):
        return np.array([self.x[i, 3 + i] for i in range(self.n)])
