"""N 台 quadrotor 用繩吊一個剛體 payload 的 MuJoCo 環境（P0.5 繩／物理 spike）。

幾何：payload 本體座標原點 = 幾何中心；N 個掛點均勻分布在半徑 r_p 的圓上（z=0）。
每機的掛點就在機體質心（不產生繩力矩——刻意簡化，減少一個變因；若之後要偏移掛點要同時改 PD 的前饋）。
payload 質量與重心偏移（body_ipos）可在執行期用 set_payload() 改。

繩模型（rope_model）：
  "tendon"（A）：spatial tendon 兩端為 quad 與 payload 掛點，limited，range=[0, L]。
      只有長度 >= L 才有約束力（單側）；鬆弛時完全沒有力、沒有阻尼。
      solref=(timeconst, dampratio)、solimp 的選值理由見 TENDON_SOLREF / TENDON_SOLIMP。
  "spring"（B）：自算單側彈簧阻尼，用 xfrc_applied 在每個物理子步施加：
      T = k*(l-L) + c*ldot（僅 l>L；c 只在拉伸時作用；T 夾在 >= 0，繩不能推）。
      預設 k=5000 N/m、c=10 N s/m，理由見 SPRING_K / SPRING_C。
      注意：xfrc_applied 在一個物理步內為常數（RK4 的四個 stage 看到同一個力），
      所以此模型的繩力是 zero-order-hold，實際積分精度低於 RK4 名義階數；隱式積分器也不會
      對這個外加力做隱式處理。這是 B 的已知代價，spike 的數據會反映。

介面（皆為 n 機獨立）：
    eta            (n,4)  各機各馬達效率（推力降效）
    ext_force      (n,3)  world frame 外力（風），作用於各機質心
    payload_ext_force (3,) 作用於 payload 質心
    set_payload(mass, com)
    step(U)        U: (n,4) 推力 (N)，回傳 dict（見 step docstring）

!!! 張力真值（rope_tension_truth、tension_peak）來自模擬器內部，只給評估／診斷用。
!!! 之後的控制器、MPPI、估計器一律不可讀取，否則違反 PLAN R1 的觀測模型（機上沒有張力計）。
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from smpc.envs.params import QuadParams

# --- 繩模型參數與理由 -------------------------------------------------------
# A：timeconst=0.02 s。MuJoCo 會把 timeconst 夾到 >= 2*timestep（refsafe），若取 0.01 在
#    physics_dt=0.005 時會被偷偷改成 0.01，使 solref 隨 dt 變動而混淆 dt 掃描；0.02 >= 4*0.005，
#    五個 dt 內都不被夾。dampratio=1（臨界阻尼）：避免約束彈回震盪，繩被拉直時不彈。
#    等效剛度 ≈ m_eff / (dmax^2 * timeconst^2) ≈ 0.3/(0.9*4e-4) ≈ 800 N/m（以實測校正，見 spike 輸出）。
TENDON_SOLREF = (0.02, 1.0)
# A：solimp=(0.9, 0.95, 0.001, 0.5, 2)。dmin=0.9：約束一啟動就有 90% 力道，不要軟啟動
#    （軟啟動會讓繩在被拉直初期可伸長很多，違背「繩幾乎不可伸長」）；dmax=0.95；width=1 mm：
#    在 1 mm 內由 dmin 過渡到 dmax，使 1 mm 量級的違反量即達最大剛度。midpoint/power 取預設。
TENDON_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
# B：k=5000 N/m。約等於 1 m 長、EA≈5 kN 的編織尼龍／聚酯細繩（直徑約 2-3 mm）；
#    更硬的 Dyneema（EA~1e5 N）會讓 omega=sqrt(k/m_eff) 到 ~600 rad/s，spike 另做 k 敏感度掃描。
#    靜態 1.5 N 時伸長 0.3 mm（0.03% L）。
SPRING_K = 5000.0
# B：c=10 N s/m。對 m_eff≈0.3 kg 阻尼比 zeta = c/(2 sqrt(k m_eff)) ≈ 0.13，合乎合成纖維繩
#    的低損耗（zeta ~0.05-0.2）；同時足以壓住拉直瞬間的高頻震盪。
SPRING_C = 10.0


@dataclass
class RopeSpec:
    model: str = "tendon"      # "tendon" | "spring"
    length: float = 1.0
    k: float = SPRING_K
    c: float = SPRING_C
    solref: tuple = TENDON_SOLREF
    solimp: tuple = TENDON_SOLIMP


def _payload_inertia(m: float, r: float, h: float = 0.04) -> np.ndarray:
    """實心圓盤（半徑 r、厚 h）繞質心的對角慣量。"""
    return np.array([m * (3 * r * r + h * h) / 12, m * (3 * r * r + h * h) / 12, 0.5 * m * r * r])


class MultiPayloadEnv:
    def __init__(self, n: int = 3, params: QuadParams | None = None, payload_mass: float = 0.4,
                 payload_com=(0.0, 0.0, 0.0), rope_length: float = 1.0, rope_model: str = "tendon",
                 dt: float = 0.01, physics_dt: float = 0.002, integrator: str = "RK4",
                 payload_radius: float = 0.15, attach_z: float = 0.0, rope_k: float = SPRING_K, rope_c: float = SPRING_C,
                 solref=TENDON_SOLREF, solimp=TENDON_SOLIMP):
        assert 2 <= n <= 6
        assert rope_model in ("tendon", "spring")
        assert integrator in ("RK4", "implicitfast", "Euler", "implicit")
        self.n, self.p = n, params or QuadParams()
        self.rope = RopeSpec(rope_model, rope_length, rope_k, rope_c, tuple(solref), tuple(solimp))
        self.L, self.r_p = rope_length, payload_radius
        self.physics_dt, self.dt = physics_dt, dt
        self.substeps = int(round(dt / physics_dt))
        assert abs(self.substeps * physics_dt - dt) < 1e-9
        ang = 2 * np.pi * np.arange(n) / n
        # 掛點（payload 本體座標）；quad 繩方向向外，故 quad 位於同方位角更外側。
        # attach_z：掛點相對幾何中心的高度（預設 0 = 與重心同高，P0.5 行為）。>0 = 從上方吊（較像真實 payload）。
        self.attach_z = float(attach_z)
        self.attach_body = np.stack([payload_radius * np.cos(ang), payload_radius * np.sin(ang),
                                     np.full(n, attach_z)], 1)
        self.azimuth = ang
        self._payload_radius = payload_radius
        self.model = mujoco.MjModel.from_xml_string(self._mjcf(integrator))
        self.data = mujoco.MjData(self.model)
        self._qb = [self.model.body(f"quad{i}").id for i in range(n)]
        self._pb = self.model.body("payload").id
        self._m0_payload = float(payload_mass)
        self.eta = np.ones((n, 4))
        self.ext_force = np.zeros((n, 3))
        self.payload_ext_force = np.zeros(3)
        self.rope_tension_truth = np.zeros(n)   # 評估用，控制器不可用
        self.tension_peak = np.zeros(n)         # 上一個控制步內各子步的最大張力，評估用
        self.set_payload(payload_mass, payload_com)
        self.reset()

    # ---------------- MJCF ----------------
    def _mjcf(self, integrator: str) -> str:
        p, n = self.p, self.n
        ix, iy, iz = p.inertia
        quads, motors, tendons = [], [], []
        for i in range(n):
            sites = "".join(f'<site name="r{i}_{j}" pos="{x} {y} 0" size="0.01"/>'
                            for j, (x, y) in enumerate(p.rotor_xy))
            quads.append(f'<body name="quad{i}" pos="0 0 1"><freejoint name="qj{i}"/>'
                         f'<inertial pos="0 0 0" mass="{p.mass}" diaginertia="{ix} {iy} {iz}"/>'
                         f'{sites}<site name="att{i}" pos="0 0 0" size="0.01"/></body>')
            for j, s in enumerate(p.rotor_spin):
                motors.append(f'<motor name="m{i}_{j}" site="r{i}_{j}" gear="0 0 1 0 0 {s * p.km}" '
                              f'ctrllimited="true" ctrlrange="0 {p.fmax}"/>')
        psites = "".join(f'<site name="pa{i}" pos="{a[0]} {a[1]} {a[2]}" size="0.01"/>'
                         for i, a in enumerate(self.attach_body))
        pi = _payload_inertia(1.0, self._payload_radius)
        payload = (f'<body name="payload" pos="0 0 0"><freejoint name="pj"/>'
                   # inertial pos 刻意給 1e-6 m：若恰為 0，MuJoCo 會把 body 編成 "simple"，之後
                   # set_payload 改 body_ipos 時 mj_setConst 會報錯。1e-6 m 的偏移物理上可忽略。
                   f'<inertial pos="1e-6 0 0" mass="1" diaginertia="{pi[0]} {pi[1]} {pi[2]}"/>{psites}</body>')
        if self.rope.model == "tendon":
            sr = " ".join(map(str, self.rope.solref))
            si = " ".join(map(str, self.rope.solimp))
            for i in range(n):
                tendons.append(f'<spatial name="rope{i}" limited="true" range="0 {self.L}" '
                               f'solreflimit="{sr}" solimplimit="{si}" width="0.003">'
                               f'<site site="att{i}"/><site site="pa{i}"/></spatial>')
        return f"""
<mujoco model="multi_payload">
  <option timestep="{self.physics_dt}" integrator="{integrator}" gravity="0 0 {-p.g}"
          solver="Newton" iterations="50" tolerance="1e-10"/>
  <worldbody>{''.join(quads)}{payload}</worldbody>
  <tendon>{''.join(tendons)}</tendon>
  <actuator>{''.join(motors)}</actuator>
</mujoco>"""

    # ---------------- payload 參數 ----------------
    def set_payload(self, mass: float, com=(0.0, 0.0, 0.0)) -> None:
        m = self.model
        m.body_mass[self._pb] = mass
        m.body_ipos[self._pb] = np.asarray(com, float)
        m.body_inertia[self._pb] = _payload_inertia(mass, self._payload_radius)
        mujoco.mj_setConst(m, self.data)
        self.payload_mass, self.payload_com = float(mass), np.asarray(com, float)

    # ---------------- 幾何 / state ----------------
    @property
    def quad_states(self) -> np.ndarray:
        """(n,13)：[pos, quat wxyz, vel(world), omega(body)]，與單機 env 同約定。"""
        d = self.data
        return np.stack([np.concatenate([d.qpos[7 * i:7 * i + 7], d.qvel[6 * i:6 * i + 6]])
                         for i in range(self.n)])

    @property
    def payload_state(self) -> np.ndarray:
        """(13,)，位置／速度為 payload 本體座標原點（幾何中心，非重心）。"""
        d, n = self.data, self.n
        return np.concatenate([d.qpos[7 * n:7 * n + 7], d.qvel[6 * n:6 * n + 6]])

    @staticmethod
    def _rot(q):
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    def _rope_kin(self):
        """(掛點世界位置 pa, 掛點速度 va, quad 位置 pq, quad 速度 vq, R_payload)。"""
        d, n = self.data, self.n
        pp, qp = d.qpos[7 * n:7 * n + 3], d.qpos[7 * n + 3:7 * n + 7]
        vp, wp = d.qvel[6 * n:6 * n + 3], d.qvel[6 * n + 3:6 * n + 6]
        R = self._rot(qp)
        pa = pp + self.attach_body @ R.T
        va = vp + (np.cross(wp, self.attach_body)) @ R.T
        pq = np.stack([d.qpos[7 * i:7 * i + 3] for i in range(n)])
        vq = np.stack([d.qvel[6 * i:6 * i + 3] for i in range(n)])
        return pa, va, pq, vq, R

    @property
    def rope_lengths(self) -> np.ndarray:
        pa, _, pq, _, _ = self._rope_kin()
        return np.linalg.norm(pq - pa, axis=1)

    def hover_positions(self, payload_pos=(0.0, 0.0, 1.0), cone_deg: float = 30.0) -> np.ndarray:
        """繩與垂直夾角 cone_deg 且繩剛好拉直的 quad 位置 (n,3)（payload 無旋轉）。"""
        th = np.deg2rad(cone_deg)
        rad = self.r_p + self.L * np.sin(th)
        off = np.stack([rad * np.cos(self.azimuth), rad * np.sin(self.azimuth),
                        np.full(self.n, self.attach_z + self.L * np.cos(th))], 1)
        return np.asarray(payload_pos, float) + off

    def reset(self, payload_pos=(0.0, 0.0, 1.0), cone_deg: float = 30.0) -> dict:
        d = self.data
        mujoco.mj_resetData(self.model, d)
        n = self.n
        qp = self.hover_positions(payload_pos, cone_deg)
        for i in range(n):
            d.qpos[7 * i:7 * i + 7] = [*qp[i], 1, 0, 0, 0]
        d.qpos[7 * n:7 * n + 7] = [*payload_pos, 1, 0, 0, 0]
        self.eta = np.ones((n, 4))
        self.ext_force = np.zeros((n, 3))
        self.payload_ext_force = np.zeros(3)
        self.rope_tension_truth = np.zeros(n)
        self.tension_peak = np.zeros(n)
        mujoco.mj_forward(self.model, d)
        return self._obs()

    def kick_payload(self, dv) -> None:
        """對 payload 加速度增量（世界座標，m/s）——造 slack 用的人為擾動。"""
        self.data.qvel[6 * self.n:6 * self.n + 3] += np.asarray(dv, float)

    # ---------------- 張力 ----------------
    def _tension_spring(self):
        pa, va, pq, vq, R = self._rope_kin()
        q = pq - pa
        l = np.linalg.norm(q, axis=1)
        u = q / l[:, None]
        s = l - self.L
        ldot = np.einsum("ij,ij->i", u, vq - va)
        T = np.where(s > 0, self.rope.k * s + self.rope.c * ldot, 0.0)
        return np.maximum(T, 0.0), u, pa, R

    def _tension_tendon(self):
        d = self.data
        T = np.zeros(self.n)
        if d.nefc:
            sel = d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON
            T[d.efc_id[:d.nefc][sel]] = d.efc_force[:d.nefc][sel]
        return T

    def _apply_spring_forces(self, base):
        T, u, pa, R = self._tension_spring()
        xf = self.data.xfrc_applied
        xf[:] = base
        com_w = self.data.qpos[7 * self.n:7 * self.n + 3] + R @ self.payload_com
        for i in range(self.n):
            f = T[i] * u[i]                       # 對 payload 的力（朝 quad）
            xf[self._qb[i], :3] -= f
            xf[self._pb, :3] += f
            xf[self._pb, 3:] += np.cross(pa[i] - com_w, f)
        return T

    # ---------------- step ----------------
    def step(self, U) -> dict:
        """施加推力 U (n,4) 一個控制週期。回傳 dict：
        quad_states (n,13)、payload_state (13,)、rope_lengths (n,)、
        tension_truth (n,)（步末）、tension_peak (n,)（步內峰值）——後兩者僅供評估，控制器不可用。"""
        d, m = self.data, self.model
        U = np.clip(np.asarray(U, float).reshape(self.n, 4), 0.0, self.p.fmax)
        d.ctrl[:] = (self.eta * U).ravel()
        base = np.zeros((m.nbody, 6))
        for i in range(self.n):
            base[self._qb[i], :3] = self.ext_force[i]
        base[self._pb, :3] = self.payload_ext_force
        peak = np.zeros(self.n)
        if self.rope.model == "spring":
            for _ in range(self.substeps):
                T = self._apply_spring_forces(base)
                peak = np.maximum(peak, T)
                mujoco.mj_step(m, d)
            self.rope_tension_truth = self._tension_spring()[0]
        else:
            d.xfrc_applied[:] = base
            for _ in range(self.substeps):
                mujoco.mj_step(m, d)
                peak = np.maximum(peak, self._tension_tendon())
            self.rope_tension_truth = self._tension_tendon()
        self.tension_peak = peak
        return self._obs()

    def _obs(self) -> dict:
        return dict(quad_states=self.quad_states, payload_state=self.payload_state,
                    rope_lengths=self.rope_lengths, tension_truth=self.rope_tension_truth.copy(),
                    tension_peak=self.tension_peak.copy())

    # ---------------- 靜力學（已知 payload 前饋用） ----------------
    def static_tensions(self, cone_deg: float = 30.0) -> np.ndarray:
        """名義幾何（payload 無旋轉、繩剛好拉直）下，已知 payload 質量／重心所需的各繩張力 (n,)。
        解 6 條 payload 力／力矩平衡式的最小二乘；n=3 時 3 個未知數 vs 6 式（對稱時自洽），
        n>3 時有多餘自由度（預張力不定），取最小範數解。這是「已知 payload」的前饋，非張力量測。"""
        qpos = self.hover_positions(np.zeros(3), cone_deg)
        a = self.attach_body
        u = qpos - a
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        A = np.zeros((6, self.n))
        A[:3] = u.T
        A[3:] = np.cross(a - self.payload_com, u).T
        b = np.array([0, 0, self.payload_mass * self.p.g, 0, 0, 0])
        T, *_ = np.linalg.lstsq(A, b, rcond=None)
        self.static_residual = float(np.linalg.norm(A @ T - b))
        return T

    def rope_unit_nominal(self, cone_deg: float = 30.0) -> np.ndarray:
        """名義幾何下 quad 指向 payload 掛點的單位向量 (n,3)。"""
        qpos = self.hover_positions(np.zeros(3), cone_deg)
        v = self.attach_body - qpos
        return v / np.linalg.norm(v, axis=1, keepdims=True)
