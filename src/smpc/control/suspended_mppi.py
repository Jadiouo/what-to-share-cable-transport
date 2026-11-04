"""分散式吊掛 MPPI（P2a-1）：每機一個 MPPI，只用自身觀測、參考、編隊幾何與 θ̂_s。

每機的預測模型（平移點質量 + 內環姿態響應）：
    p' = v,  v' = a_act + F_rope/m + d_loc/m,   (a_act, a_act') 為二階低通（內環 SO(3) PD 的近似）
    a_act'' = wn^2 (u - a_act) - 2 z wn a_act'
  u = 規劃的「推力加速度指令」(= F_cmd/m - g e_z)，直接餵 GeometricPD.wrench 的 acc_d（位置迴路增益歸零），
  所以姿態迴路與 mixer 是既有程式；MPPI 只決定 u。wn、z 由階躍響應校正（pilot 階段）。
  整個預測模型是線性時不變（對 [p,v,a,a']），用精確離散化（ZOH）。

繩力前饋 F_rope,i(k)（PLAN R1：不從量測回算張力）：
    由 θ̂_s=(δm, δc_x, δc_y)、payload 參考加速度 a_ref(t_k)、名義編隊幾何，解 payload 的力／力矩平衡
        Σ T_i w_i = (m0+δm)(a_ref + g e_z),   Σ (r_i - c) x T_i w_i = 0   (w_i = 繩由 payload 指向機 i)
    的加權最小二乘（力權重 1、力矩權重 w_moment），得 T̂_i(k)；繩力方向取自己量到的繩方向 q̃_i
    （自身感測；沒有就用名義）。機 i 受力 = T̂_i * q̃_i（q̃ 指向 payload 掛點）。
  P2a-2 換估計器：只要每步呼叫 set_theta(i, θ̂)（以及可選的 set_local_residual(i, f)），控制器本體不變。

成本（每機獨立，不含任何 payload 真值）：
    位置／速度追蹤（對自己的編隊參考點）+ ‖δ‖²（偏離前饋的量）+ 推力上限／傾角上限的軟懲罰
    + 繩近 slack 懲罰（機到參考掛點距離 < L - margin：用自己的位置與參考，不依賴張力）。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from smpc.control.mppi import MPPI
from smpc.control.pd import GeometricPD


@dataclass
class MPPIConfig:
    K: int = 512
    H: int = 20
    dt_plan: float = 0.05
    lam: float = 1.0
    sigma: float = 1.0           # δ 的取樣標準差 (m/s^2)
    noise_corr: float = 0.8      # 取樣噪聲時間相關 ρ（AR(1)）：獨立噪聲湊不出持續的大修正（pilot 階段的觀察）
    q_p: float = 100.0
    q_v: float = 10.0
    q_pT: float = 400.0          # 終端位置權重
    q_vT: float = 40.0
    r_u: float = 0.02
    thrust_lim: float = 24.0     # N（單機 4*fmax=32，留裕度）
    tilt_lim_deg: float = 35.0
    w_lim: float = 500.0
    w_slack: float = 300.0
    slack_margin: float = 0.03
    wn: float = 12.2             # 內環水平響應（姿態迴路，由階躍校正）
    zeta: float = 0.82
    wn_z: float = 40.0           # 垂直（總推力）響應：幾乎瞬時
    zeta_z: float = 1.0
    w_moment: float = 1.0
    use_measured_dir: bool = True
    cone_deg: float = 30.0

    @staticmethod
    def load(path, **over):
        d = json.load(open(path))
        d = d.get("mppi", d)
        d.update(over)
        return MPPIConfig(**{k: v for k, v in d.items() if k in MPPIConfig.__dataclass_fields__})


def _discretize(dt, wn, z):
    """每軸狀態 [p,v,a,a']，輸入 u（推力加速度指令）與 d（繩力加速度，加在 v'）。回傳 Φ(4x4), Γu(4), Γd(4)。"""
    A = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, -wn ** 2, -2 * z * wn]], float)
    B = np.array([0, 0, 0, wn ** 2], float)
    E = np.array([0, 1, 0, 0], float)
    M = np.zeros((6, 6)); M[:4, :4] = A; M[:4, 4] = B; M[:4, 5] = E
    Md = torch.linalg.matrix_exp(torch.as_tensor(M * dt, dtype=torch.float64)).numpy()
    return Md[:4, :4], Md[:4, 4], Md[:4, 5]


def _expand(dt, cfg):
    """把每軸 4 態展開成 [p(3),v(3),a(3),a'(3)] 的 12 維（x、y 同一組參數，z 另一組）；
    回傳 (12,12)^T 與 (3,12) 形式供 x @ M 使用。"""
    P = np.zeros((12, 12)); Bu = np.zeros((3, 12)); Bd = np.zeros((3, 12))
    for ax in range(3):
        wn, z = (cfg.wn, cfg.zeta) if ax < 2 else (cfg.wn_z, cfg.zeta_z)
        Phi, Gu, Gd = _discretize(dt, wn, z)
        for i in range(4):
            for j in range(4):
                P[3 * i + ax, 3 * j + ax] = Phi[i, j]
            Bu[ax, 3 * i + ax] = Gu[i]
            Bd[ax, 3 * i + ax] = Gd[i]
    return P.T.copy(), Bu, Bd


class AgentMPPI:
    """第 i 台機的規劃器。只持有：自身幾何（掛點、名義繩方向）、θ̂、本地殘差、參考函式。"""

    def __init__(self, i, attach_body, form_offset, rope_L, traj, params, cfg: MPPIConfig, m_nom_payload,
                 seed=0):
        self.i, self.cfg, self.p, self.traj = i, cfg, params, traj
        self.m_q = params.mass
        self.g = params.g
        self.L = rope_L
        self.r = np.asarray(attach_body, float)       # 本機掛點（payload 座標），名義
        self.off = np.asarray(form_offset, float)      # 編隊：quad 位置 = payload 位置 + off
        self.m0 = m_nom_payload
        self.theta = np.zeros(3)                       # (δm, δc_x, δc_y)
        self.d_loc = np.zeros(3)                       # 本地殘差力（P2a-1 先 0）
        self.eta_hat = 1.0                             # P2b：本地推力效率估計 η̂；指令除以 η̂ 以補償降效（1 = 不補償，行為同 P2a）
        self.a_bias = None                             # P2c 診斷鉤子：前饋張力用的加速度 = a_ref + a_bias（預設 None = 不變）
        self.tension_fn = None                         # P2b：不通訊基線（DOB-hold / LOADSHARE）覆寫繩力前饋：fn(t, ts, a_ref, q) -> F_rope (H+1,3)
        self.u_nom_dir = (self.r - self.off) / np.linalg.norm(self.r - self.off)  # 名義：quad->掛點
        self.x_inner = np.zeros(12)                    # 內環 [p,v,a,a'] 中 a,a' 的內部估計（p,v 由量測覆寫）
        self.pd = GeometricPD(params)
        P, Bu, Bd = _expand(cfg.dt_plan, cfg)
        t = lambda a: torch.as_tensor(a, dtype=torch.float32)
        self.Pt, self.Bu, self.Bd = t(P), t(Bu), t(Bd)
        self._inner = _expand(0.01, cfg)[:2]           # 控制率（0.01 s）內環估計用（沒有量測時的備案）
        self._thrust_last = params.mass * params.g     # 起始假設懸停推力
        self._a_c_last = None
        self.mppi = MPPI(self._dyn, self._cost, nx=12, nu=3, K=cfg.K, H=cfg.H, lam=cfg.lam, sigma=cfg.sigma, noise_corr=cfg.noise_corr,
                         terminal=self._term, traj_cost=self._tcost, seed=seed)
        self.g_vec = np.array([0, 0, -self.g])
        self._tilt_tan = float(np.tan(np.deg2rad(cfg.tilt_lim_deg)))
        self._build_P()
        self.last_plan_time = 0.0

    # ---------- θ̂ 介面（估計器只動這裡） ----------
    def set_theta(self, theta):
        self.theta = np.asarray(theta, float).copy()
        self._build_P()

    def set_local_residual(self, f):
        self.d_loc = np.asarray(f, float).copy()

    def set_eta(self, eta):
        """P2b：本地推力效率 η̂（真值或本地估計）。act() 回傳的 rotor 指令 = 規劃推力 / η̂。"""
        self.eta_hat = float(np.clip(eta, 0.4, 1.2))

    def _build_P(self):
        """名義編隊 + θ̂ 下，力需求 (3,) -> 本機張力 T_i 的線性映射 P_i (3,)。N 機聯立的 lstsq 需要全隊幾何，
        所以由 Team 在建構／set_theta 時呼叫 set_tension_map；這裡只佔位。"""
        pass

    def set_tension_map(self, P_i):
        self.P_i = np.asarray(P_i, float)

    # ---------- 前饋 ----------
    def _feedforward(self, t, rope_dir):
        cfg, H = self.cfg, self.cfg.H
        ts = t + cfg.dt_plan * np.arange(H + 1)
        p, v, a = self.traj.ref(ts)                       # (H+1,3)
        a_T = a if self.a_bias is None else a + np.asarray(self.a_bias)[None]
        f_req = (self.m0 + self.theta[0]) * (a_T + np.array([0, 0, self.g]))
        T = f_req @ self.P_i                              # (H+1,)
        T = np.maximum(T, 0.0)
        q = rope_dir if (cfg.use_measured_dir and rope_dir is not None) else self.u_nom_dir
        if self.tension_fn is not None:                   # P2b 不通訊基線：直接給繩力向量序列
            F_rope = np.asarray(self.tension_fn(t, ts, a, q), float)
            T = np.maximum(F_rope @ q, 0.0)
        else:
            F_rope = T[:, None] * q[None]                 # 繩對本機的力（朝 payload）
        self.last_T = T
        F_rope = F_rope + self.d_loc[None]
        d = F_rope / self.m_q                             # 加速度
        a_ff = a - d                                      # 推力加速度的前饋 (a_c = a_ref - F_rope/m)
        pq = p + self.off[None]
        vq = v
        pa = p + self.r[None]                             # 參考掛點
        return pq, vq, d, a_ff, pa, T

    def _dyn(self, x, u, k):
        ut = u + self._a_ff[k]
        return x @ self.Pt + ut @ self.Bu + self._d[k] @ self.Bd

    def _cost(self, x, u, k):
        c = self.cfg
        j = k + 1
        ep = x[:, 0:3] - self._pq[j]
        ev = x[:, 3:6] - self._vq[j]
        S = c.q_p * (ep ** 2).sum(1) + c.q_v * (ev ** 2).sum(1) + c.r_u * (u ** 2).sum(1)
        ut = u + self._a_ff[k]
        F = self.m_q * (ut + self._gup)
        Fn = F.norm(dim=1)
        S = S + c.w_lim * torch.relu(Fn - c.thrust_lim) ** 2
        Fh = F[:, :2].norm(dim=1)
        S = S + c.w_lim * torch.relu(Fh - self._tilt_tan * F[:, 2].clamp(min=0.1)) ** 2
        dist = (x[:, 0:3] - self._pa[j]).norm(dim=1)
        S = S + c.w_slack * torch.relu(self.L - c.slack_margin - dist) ** 2
        return S

    def _tcost(self, X, U):
        """與 _cost 逐步相加等價的向量化版本。X: (K,H,12)，U: (K,H,3)。"""
        c, H = self.cfg, self.cfg.H
        ep = X[..., 0:3] - self._pq[1:H + 1]
        ev = X[..., 3:6] - self._vq[1:H + 1]
        S = c.q_p * (ep ** 2).sum(-1) + c.q_v * (ev ** 2).sum(-1) + c.r_u * (U ** 2).sum(-1)
        F = self.m_q * (U + self._a_ff[:H] + self._gup)
        Fn = F.norm(dim=-1)
        S = S + c.w_lim * torch.relu(Fn - c.thrust_lim) ** 2
        Fh = F[..., :2].norm(dim=-1)
        S = S + c.w_lim * torch.relu(Fh - self._tilt_tan * F[..., 2].clamp(min=0.1)) ** 2
        dist = (X[..., 0:3] - self._pa[1:H + 1]).norm(dim=-1)
        S = S + c.w_slack * torch.relu(self.L - c.slack_margin - dist) ** 2
        return S.sum(-1)

    def _term(self, x):
        c = self.cfg
        H = c.H
        ep = x[:, 0:3] - self._pq[H]
        ev = x[:, 3:6] - self._vq[H]
        return c.q_pT * (ep ** 2).sum(1) + c.q_vT * (ev ** 2).sum(1)

    def _update_inner(self, state):
        """內環狀態 (a, a') 的估計：a = 上一步實際推力總和 * 目前機體 z 軸 / m + g（只用自身姿態與自己送出的推力，
        等同 IMU 加上推力指令的組合；馬達降效時此處會偏，那屬於本地殘差的範圍）；a' 用 a 的濾波差分。"""
        from smpc.control.pd import quat_to_rot
        b3 = quat_to_rot(state[3:7])[:, 2]
        a = self._thrust_last / self.m_q * b3 + self.g_vec
        if self._a_c_last is None:
            ad = np.zeros(3)
        else:                                   # a' 取模型預測（用上一步已套用的指令推進一步），a 取量測
            Pm, Bm = self._inner
            xi = np.concatenate([np.zeros(6), self.x_inner[6:12]]) @ Pm + self._a_c_last @ Bm
            ad = xi[9:12]
        self.x_inner[6:9], self.x_inner[9:12] = a, ad

    # ---------- 每個控制步 ----------
    def act(self, t, state, rope_dir=None):
        """state: 本機 13 維狀態；rope_dir: 自身量到的繩方向（機 -> 掛點單位向量）或 None。回傳四個 rotor 推力。"""
        t0 = time.thread_time()
        pq, vq, d, a_ff, pa, self.last_T = self._feedforward(t, rope_dir)
        T = lambda a: torch.as_tensor(a, dtype=torch.float32)
        self._pq, self._vq, self._d, self._a_ff, self._pa = T(pq), T(vq), T(d), T(a_ff), T(pa)
        self._gup = T(np.array([0, 0, self.g]))
        self._update_inner(state)
        x0 = np.concatenate([state[0:3], state[7:10], self.x_inner[6:9], self.x_inner[9:12]])
        du = self.mppi.plan(x0)
        a_c = a_ff[0] + du                                # 總推力加速度指令
        T_, tau = self.pd.wrench(state, state[0:3], state[7:10], acc_d=a_c)
        u = self.pd.mix(T_, tau)
        if self.eta_hat != 1.0:
            u = np.minimum(u, self.eta_hat * self.p.fmax)  # 降效後單顆實際上限
        self._thrust_last = float(u.sum())                # 有效推力（補償後 = 規劃值）
        self._a_c_last = a_c
        self.last_plan_time = time.thread_time() - t0
        return u / self.eta_hat


class SuspendedMPPITeam:
    """N 台各自一個 AgentMPPI 的集合。沒有任何跨機的資訊流（只共享「編隊幾何與參考軌跡」這種事前約定）。"""

    def __init__(self, env, traj, cfg: MPPIConfig, theta_hat=None, m_nom=1.0, seed=0):
        self.env, self.traj, self.cfg = env, traj, cfg
        n = env.n
        pos0 = env.hover_positions(np.zeros(3), cfg.cone_deg)
        self.offsets = pos0
        self.m_nom = m_nom   # 名義 payload 質量（控制器已知）；真值 = m_nom + δm
        self.agents = [AgentMPPI(i, env.attach_body[i], pos0[i], env.L, traj, env.p, cfg, self.m_nom,
                                 seed=seed * 100 + i) for i in range(n)]
        self.theta = np.zeros((n, 3))
        self.set_theta_all(np.zeros(3) if theta_hat is None else theta_hat)

    def set_nominal_mass(self, m):
        for a in self.agents:
            a.m0 = m

    def _tension_maps(self, c):
        """名義幾何 + 重心偏移 c=(cx,cy) 下的力需求->各繩張力線性映射 (n,3)。
        注意：各機若有各自的 θ̂（分散式），各自用自己的 c 來解——此處對第 i 機傳入其 θ̂ 的 c。"""
        env, cfg = self.env, self.cfg
        n = env.n
        wq = -(env.attach_body - self.offsets)                                       # = quad - 掛點 方向（payload->quad 未正規化）
        wq = wq / np.linalg.norm(wq, axis=1, keepdims=True)                         # 單位向量 payload -> quad
        cv = np.array([c[0], c[1], 0.0])
        A = np.zeros((6, n))
        A[:3] = wq.T
        A[3:] = np.cross(env.attach_body - cv, wq).T
        W = np.diag([1, 1, 1] + [cfg.w_moment] * 3)
        P = np.linalg.pinv(W @ A)[:, :3]     # b=[F;0] -> T = pinv(WA) (W b)，W 對 F 部分為 1
        return P                              # (n,3)

    def set_theta(self, i, theta):
        self.theta[i] = theta
        a = self.agents[i]
        a.set_theta(theta)
        a.set_tension_map(self._tension_maps(theta[1:3])[i])

    def set_eta(self, i, eta):
        self.agents[i].set_eta(eta)

    def set_tension_fn(self, i, fn):
        self.agents[i].tension_fn = fn

    def set_theta_all(self, theta):
        theta = np.asarray(theta, float)
        for i in range(self.env.n):
            self.set_theta(i, theta)

    def act(self, t, quad_states, rope_dirs=None):
        return np.stack([self.agents[i].act(t, quad_states[i], None if rope_dirs is None else rope_dirs[i])
                         for i in range(self.env.n)])

    @property
    def step_time(self):
        """上一步各機的計算時間 (n,)"""
        return np.array([a.last_plan_time for a in self.agents])

    def feedforward_tension(self):
        return np.array([a.last_T[0] for a in self.agents])
