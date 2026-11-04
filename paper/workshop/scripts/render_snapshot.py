"""Render a simulator snapshot for the paper (Fig. 1).

Step 1 (CPU): rerun one formal episode (CON-ATT, s=1, circle, seed 3000, frozen hp) and record qpos.
  The RMSE is checked against results/prereg/episodes.jsonl so we know the recorded run is the formal one.
Step 2 (CPU software GL, Mesa llvmpipe; no GPU): add visual-only geoms to the same MJCF and render chosen frames.

  .venv/bin/python paper/workshop/scripts/render_snapshot.py record
  __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json LIBGL_ALWAYS_SOFTWARE=1 \
    .venv/bin/python paper/workshop/scripts/render_snapshot.py render --t 2.0 4.0 6.0
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REC = ROOT / ".scratch/snapshot_qpos.npz"
OUT = ROOT / "paper/workshop/figs"
METHOD, PROFILE, TRAJ, SEED = "CON-ATT", "s=1", "circle", 3000
HP = {"lam": 0.9998, "prior": 2.0, "iters": 2, "p_drop": 0.0}   # experiments/prereg_run.py HP["CON-ATT"]


def record():
    import torch
    torch.set_num_threads(1)
    from smpc.envs.multi_payload import MultiPayloadEnv
    from smpc.eval.episode_p2c import run_episode_p2c
    log = {}
    orig = MultiPayloadEnv.step

    def step(self, U):   # 只記錄，不改動模擬
        o = orig(self, U)
        log.setdefault("qpos", []).append(self.data.qpos.copy())
        log["mjcf"] = self._mjcf("RK4")
        return o
    MultiPayloadEnv.step = step
    r = run_episode_p2c(METHOD, TRAJ, SEED, PROFILE, 1.0, dict(HP), eta_comp="est", n=3)
    ref = None
    for line in open(ROOT / "results/prereg/episodes.jsonl"):
        j = json.loads(line)
        if (j["method"], j["profile"], j["traj"], j["seed"], j["speed"]) == (METHOD, PROFILE, TRAJ, SEED, 1.0) \
                and j["hp"] == HP:
            ref = j["rmse"]
    print(f"rerun rmse {r['rmse']:.6f}  formal rmse {ref:.6f}  match {abs(r['rmse'] - ref) < 1e-9}")
    from smpc.control.trajectories import Traj
    tr = Traj(TRAJ, speed=1.0)
    ref_path = np.array([tr.ref(t)[0][0] for t in np.linspace(0, 10, 400)])
    np.savez(REC, qpos=np.array(log["qpos"]), mjcf=log["mjcf"], ref_path=ref_path, rmse=r["rmse"])


def visual_mjcf(mjcf, ref_path, n=3):
    from smpc.envs.params import QuadParams
    p = QuadParams()
    d = p.arm / np.sqrt(2.0)
    quad_vis = (f'<geom type="box" size="{p.arm} 0.008 0.006" euler="0 0 45" rgba=".15 .15 .18 1" {NC}/>'
                f'<geom type="box" size="{p.arm} 0.008 0.006" euler="0 0 -45" rgba=".15 .15 .18 1" {NC}/>'
                f'<geom type="box" size="0.045 0.045 0.02" rgba=".85 .35 .1 1" {NC}/>'
                + "".join(f'<geom type="cylinder" pos="{x} {y} 0.012" size="0.075 0.003" rgba=".55 .75 .95 .55" {NC}/>'
                          f'<geom type="cylinder" pos="{x} {y} 0.004" size="0.012 0.01" rgba=".1 .1 .1 1" {NC}/>'
                          for x, y in p.rotor_xy))
    for i in range(n):
        mjcf = mjcf.replace(f'<site name="att{i}"', quad_vis + f'<site name="att{i}"', 1)
    mjcf = mjcf.replace('<site name="pa0"', f'<geom type="cylinder" size="0.15 0.03" rgba=".2 .45 .75 1" {NC}/><site name="pa0"', 1)
    mjcf = re.sub(r'width="0.003"', 'width="0.004" rgba=".1 .1 .1 1"', mjcf)
    dots = "".join(f'<site pos="{x:.4f} {y:.4f} {z:.4f}" size="0.008" rgba=".8 .1 .1 .8"/>' for x, y, z in ref_path[::4])
    world = ('<light pos="0 0 6" dir="0 0 -1" diffuse=".8 .8 .8" castshadow="true"/>'
             '<geom type="plane" size="4 4 .1" material="grid" contype="0" conaffinity="0"/>' + dots)
    mjcf = mjcf.replace("<worldbody>", "<worldbody>" + world, 1)
    assets = ('<visual><global offwidth="1800" offheight="1200"/><quality shadowsize="4096"/>'
              '<headlight ambient=".4 .4 .4" diffuse=".5 .5 .5"/></visual>'
              '<asset><texture name="g" type="2d" builtin="checker" rgb1=".93 .93 .93" rgb2=".86 .86 .86" '
              'width="512" height="512"/><material name="grid" texture="g" texrepeat="8 8"/>'
              '<texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="16" height="16"/></asset>')
    return mjcf.replace("<worldbody>", assets + "<worldbody>", 1)


NC = 'contype="0" conaffinity="0" mass="0"'


def sw_context():
    """Headless Mesa surfaceless EGL context (CPU llvmpipe), so rendering never touches the GPU."""
    import ctypes
    import os
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    from OpenGL import EGL, GL
    get = ctypes.CFUNCTYPE(EGL.EGLDisplay, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p)(
        EGL.eglGetProcAddress(b"eglGetPlatformDisplayEXT"))
    dpy = get(0x31DD, None, None)   # EGL_PLATFORM_SURFACELESS_MESA
    assert EGL.eglInitialize(dpy, ctypes.pointer(EGL.EGLint()), ctypes.pointer(EGL.EGLint()))
    attrs = (EGL.EGLint * 11)(EGL.EGL_RED_SIZE, 8, EGL.EGL_GREEN_SIZE, 8, EGL.EGL_BLUE_SIZE, 8, EGL.EGL_DEPTH_SIZE, 24,
                              EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT, EGL.EGL_NONE)
    cfg, n = EGL.EGLConfig(), EGL.EGLint()
    EGL.eglChooseConfig(dpy, attrs, ctypes.pointer(cfg), 1, ctypes.pointer(n))
    EGL.eglBindAPI(EGL.EGL_OPENGL_API)
    ctx = EGL.eglCreateContext(dpy, cfg, EGL.EGL_NO_CONTEXT, None)
    assert EGL.eglMakeCurrent(dpy, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, ctx)
    renderer = GL.glGetString(GL.GL_RENDERER).decode()
    assert "llvmpipe" in renderer, renderer
    return renderer


def render(times, az, el, dist):
    import mujoco
    from PIL import Image
    print("GL renderer:", sw_context())
    z = np.load(REC)
    m = mujoco.MjModel.from_xml_string(visual_mjcf(str(z["mjcf"]), z["ref_path"]))
    dat = mujoco.MjData(m)
    W, H = 1800, 1200
    scn, opt = mujoco.MjvScene(m, maxgeom=2000), mujoco.MjvOption()
    con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, con)
    vp = mujoco.MjrRect(0, 0, W, H)
    rgb = np.zeros((H, W, 3), np.uint8)
    cam = mujoco.MjvCamera()
    for t in times:
        k = int(round(t / 0.01)) - 1
        dat.qpos[:] = z["qpos"][k]
        mujoco.mj_forward(m, dat)
        cam.lookat[:] = 0.5 * (dat.qpos[21:24] + z["ref_path"].mean(0)) + np.array([0, 0, 0.65])   # payload 與參考圓心之間
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        mujoco.mjv_updateScene(m, dat, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(vp, scn, con)
        mujoco.mjr_readPixels(rgb, None, vp, con)
        f = OUT / f"sim_snapshot_t{t:g}.png"
        img = np.flipud(rgb)
        rows = np.where((img < 250).any(axis=(1, 2)))[0]   # 裁掉上方全白的天空
        Image.fromarray(img[max(rows[0] - 40, 0):]).save(f)
        print("wrote", f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "render"])
    ap.add_argument("--t", type=float, nargs="+", default=[4.0])
    ap.add_argument("--az", type=float, default=60)
    ap.add_argument("--el", type=float, default=-16)
    ap.add_argument("--dist", type=float, default=2.8)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    record() if a.mode == "record" else render(a.t, a.az, a.el, a.dist)
