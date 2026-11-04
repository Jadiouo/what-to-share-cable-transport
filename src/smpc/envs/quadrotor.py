"""單機 quadrotor 的 MuJoCo 環境。

State 約定（長度 13 的 numpy 陣列）：
    [0:3]   位置        world frame (m)
    [3:7]   四元數      (w, x, y, z)，body -> world
    [7:10]  線速度      world frame (m/s)
    [10:13] 角速度      body frame (rad/s)
這剛好就是 MuJoCo free joint 的 qpos/qvel 約定（free joint 的角速度本來就在 body frame），
所以 state 與 qpos/qvel 之間不需要任何轉換，少一個出錯的地方。

動作 u：四個 rotor 的推力 (N)，沿 body +z，會被夾在 [0, fmax]。
反扭矩 = spin * km * 推力（見 params.allocation）。

積分器與時間步：
    MuJoCo RK4，physics_dt = 0.002 s (500 Hz)，控制週期 dt 預設 0.01 s（每步 5 個子步，
    推力在子步間保持不變 = zero-order hold）。
    - RK4：自由飛行沒有接觸，RK4 四階精度、不像 semi-implicit Euler 那樣有數值阻尼，
      自由落體與懸停的漂移測試才量得到真正的物理而非積分誤差。
    - 0.002 s：之後掛繩子用的是軟約束（solref），繩的拉緊／鬆弛頻率遠高於機體動力學，
      先用小步長留餘裕；P0.5 的 spike 再量最大可用 dt。
    - 控制週期與物理步長分開：MPPI 之後可能想用更大的控制週期，不必動物理設定。

擴充鉤子（之後多機吊掛／適應估計會用）：
    eta           shape (4,)，每馬達效率，實際推力 = eta * clip(u)。
                  放在夾限之後，所以 η<1 代表「同樣指令出力變少」，上限也跟著降低。
    ext_force     world frame 外力 (N)，作用於質心（寫入 xfrc_applied）。
    ext_torque    world frame 外力矩 (N m)。
    set_mass(m)   改機體質量，慣量預設等比縮放（想獨立改慣量可 scale_inertia=False）。
"""
from __future__ import annotations

import mujoco
import numpy as np

from smpc.envs.params import QuadParams


def build_mjcf(p: QuadParams, physics_dt: float) -> str:
    """產生 MJCF 字串。慣量以 <inertial> 明確指定，不讓 MuJoCo 從 geom 推，避免偷偷改掉質量。"""
    ix, iy, iz = p.inertia
    sites, motors = [], []
    for i, ((x, y), s) in enumerate(zip(p.rotor_xy, p.rotor_spin)):
        sites.append(f'<site name="rotor{i}" pos="{x} {y} 0" size="0.01"/>')
        # gear = [fx fy fz tx ty tz]（site frame）：推力沿 +z，並附帶繞 z 的反扭矩。
        motors.append(
            f'<motor name="m{i}" site="rotor{i}" gear="0 0 1 0 0 {s * p.km}" '
            f'ctrllimited="true" ctrlrange="0 {p.fmax}"/>'
        )
    # 視覺用 geom：contype=0 不參與碰撞，且 mass=0 不影響質量。
    arms = "".join(
        f'<geom type="capsule" fromto="0 0 0 {x} {y} 0" size="0.008" '
        f'contype="0" conaffinity="0" mass="0" rgba="0.2 0.2 0.2 1"/>'
        for x, y in p.rotor_xy
    )
    return f"""
<mujoco model="quadrotor">
  <option timestep="{physics_dt}" integrator="RK4" gravity="0 0 {-p.g}"/>
  <worldbody>
    <body name="quad" pos="0 0 1">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="{p.mass}" diaginertia="{ix} {iy} {iz}"/>
      {arms}
      {''.join(sites)}
    </body>
  </worldbody>
  <actuator>{''.join(motors)}</actuator>
</mujoco>"""


class QuadrotorEnv:
    def __init__(self, params: QuadParams | None = None, dt: float = 0.01,
                 physics_dt: float = 0.002):
        self.params = params or QuadParams()
        self.physics_dt = physics_dt
        self.substeps = int(round(dt / physics_dt))
        assert abs(self.substeps * physics_dt - dt) < 1e-9, "dt 必須是 physics_dt 的整數倍"
        self.dt = dt
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.params, physics_dt))
        self.data = mujoco.MjData(self.model)
        self._body = self.model.body("quad").id
        self._nominal_mass = self.model.body_mass[self._body]
        self._nominal_inertia = self.model.body_inertia[self._body].copy()
        self.eta = np.ones(4)
        self.ext_force = np.zeros(3)
        self.ext_torque = np.zeros(3)
        self.reset()

    # ---- state 讀寫 ----
    @property
    def state(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel])

    @property
    def mass(self) -> float:
        return float(self.model.body_mass[self._body])

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        """重置。state=None 時回到 (0,0,1) 靜止。鉤子（eta、外力、質量）一律還原成名義值，
        避免上一個 episode 的擾動洩漏到下一個。"""
        mujoco.mj_resetData(self.model, self.data)
        if state is None:
            state = np.array([0, 0, 1.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        state = np.asarray(state, dtype=float)
        self.data.qpos[:] = state[:7]
        self.data.qpos[3:7] /= np.linalg.norm(self.data.qpos[3:7])
        self.data.qvel[:] = state[7:]
        self.eta = np.ones(4)
        self.ext_force = np.zeros(3)
        self.ext_torque = np.zeros(3)
        self.set_mass(self._nominal_mass)
        mujoco.mj_forward(self.model, self.data)
        return self.state

    def set_mass(self, mass: float, scale_inertia: bool = True) -> None:
        self.model.body_mass[self._body] = mass
        scale = mass / self._nominal_mass if scale_inertia else 1.0
        self.model.body_inertia[self._body] = self._nominal_inertia * scale
        mujoco.mj_setConst(self.model, self.data)

    def step(self, u) -> np.ndarray:
        """施加四個 rotor 推力 u (N) 一個控制週期，回傳新 state。"""
        u = np.clip(np.asarray(u, dtype=float), 0.0, self.params.fmax)
        self.data.ctrl[:] = self.eta * u
        # xfrc_applied 的前 3 個是力、後 3 個是力矩，皆 world frame、作用於質心。
        self.data.xfrc_applied[self._body, :3] = self.ext_force
        self.data.xfrc_applied[self._body, 3:] = self.ext_torque
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        return self.state
