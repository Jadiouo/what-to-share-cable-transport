"""Honest demo videos for the pre-registered study (MuJoCo 3D + error plot), CPU only.

Episodes are chosen by rules fixed in advance (results/media/README.md), using only numbers in
results/prereg/episodes.jsonl. Every shown episode is re-run (monkeypatched MultiPayloadEnv.step records qpos)
and its RMSE must equal the formal one to 1e-9, otherwise no video is produced.

Run from the project root (CPU only, software GL):
  systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 \
    env CUDA_VISIBLE_DEVICES= __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
    LIBGL_ALWAYS_SOFTWARE=1 .venv/bin/python experiments/render_videos.py {select|record|render|all} [--only NAME ...]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCR = ROOT / ".scratch/render_videos"
OUT = ROOT / "results/media"
EP = ROOT / "results/prereg/episodes.jsonl"
SEEDS = list(range(3000, 3040))
DT = 0.01
STRIDE = 4          # 25 fps
HOLD_S = 1.0
W, H = 1080, 720
TOP_H = 440         # height of the 3D region (px); the error plot takes the rest

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from prereg_run import HP  # noqa: E402  frozen hp


def _load_snapshot():
    spec = importlib.util.spec_from_file_location("render_snapshot", ROOT / "paper/workshop/scripts/render_snapshot.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- data / selection
def load_eps():
    D = {}
    for line in open(EP):
        if not line.strip():
            continue
        j = json.loads(line)
        D[(j["method"], j["profile"], j["traj"], j["speed"], j["seed"], json.dumps(j["hp"], sort_keys=True))] = j
    return D


def get(D, method, prof, traj, seed):
    k = (method, prof, traj, 1.0, seed, json.dumps(HP[method], sort_keys=True))
    assert k in D, k
    return D[k]


def series(D, method, prof, traj, key="rmse"):
    return np.array([get(D, method, prof, traj, s)[key] for s in SEEDS])


def pick_gap(D, left, right, prof, traj):
    a, b = series(D, left, prof, traj), series(D, right, prof, traj)
    gap = a - b
    i = int(np.argmin(np.abs(gap - gap.mean())))   # argmin returns the first (smallest seed) on ties
    return dict(seed=SEEDS[i], gap=float(gap[i]), mean_gap=float(gap.mean()),
                rmse_l=float(a[i]), rmse_r=float(b[i]), mean_l=float(a.mean()), mean_r=float(b.mean()))


def selection(D):
    S = {}
    S["hero_con_s1"] = dict(left="CON-ATT", right=None, prof="s=1", traj="circle", seed=3000,
                            rmse_l=get(D, "CON-ATT", "s=1", "circle", 3000)["rmse"],
                            mean_l=float(series(D, "CON-ATT", "s=1", "circle").mean()))
    # 2: DOB failures at s=2.5
    traj, note = "circle", ""
    r = series(D, "DOB", "s=2.5", traj)
    ok = series(D, "DOB", "s=2.5", traj, "success")
    fail = [i for i in range(40) if not ok[i]]
    if not fail:
        traj, note = "fig8", "no DOB failure on circle at s=2.5; fell back to fig8"
        r = series(D, "DOB", "s=2.5", traj)
        ok = series(D, "DOB", "s=2.5", traj, "success")
        fail = [i for i in range(40) if not ok[i]]
    fr = r[fail]
    med = float(np.median(fr))
    j = min(range(len(fail)), key=lambda q: (abs(fr[q] - med), fail[q]))   # closest to the median; ties -> smaller seed
    i = fail[j]
    c = series(D, "CON-ATT", "s=2.5", traj)
    S["dob_failure_s25"] = dict(left="DOB", right="CON-ATT", prof="s=2.5", traj=traj, seed=SEEDS[i],
                                rmse_l=float(r[i]), rmse_r=float(c[i]), mean_l=float(r.mean()), mean_r=float(c.mean()),
                                n_fail=len(fail), fail_median=med, fail_seeds=[SEEDS[q] for q in fail], note=note)
    g = pick_gap(D, "EKF-L", "EKF-L+GEO", "s=1.75", "circle")
    S["geo_vs_local_s175"] = dict(left="EKF-L", right="EKF-L+GEO", prof="s=1.75", traj="circle", **g)
    g = pick_gap(D, "EKF-L+GEO", "CON-ATT", "s=0", "circle")
    S["con_vs_geo_s0"] = dict(left="EKF-L+GEO", right="CON-ATT", prof="s=0", traj="circle", **g)
    return S


def episodes_needed(S):
    need = []
    for v in S.values():
        for m in (v["left"], v.get("right")):
            if m:
                e = (m, v["prof"], v["traj"], v["seed"])
                if e not in need:
                    need.append(e)
    return need


def cache_path(e):
    return SCR / f"qpos_{e[0].replace('+', 'p')}_{e[1]}_{e[2]}_{e[3]}.npz"


# ---------------------------------------------------------------- record (rerun + verify)
def record(S, D):
    import torch
    torch.set_num_threads(1)
    from smpc.control.trajectories import Traj
    from smpc.envs.multi_payload import MultiPayloadEnv
    from smpc.eval.episode_p2c import run_episode_p2c
    SCR.mkdir(parents=True, exist_ok=True)
    log = {}
    orig = MultiPayloadEnv.step

    def step(self, U):   # record only; the simulation is untouched
        o = orig(self, U)
        log["qpos"].append(self.data.qpos.copy())
        if "mjcf" not in log:
            log["mjcf"] = self._mjcf("RK4")
        return o
    MultiPayloadEnv.step = step
    bad = []
    for e in episodes_needed(S):
        m, prof, traj, seed = e
        log.clear()
        log["qpos"] = []
        r = run_episode_p2c(m, traj, seed, prof, 1.0, dict(HP[m]), eta_comp=_eta(m), n=3)
        f = get(D, m, prof, traj, seed)
        d = abs(r["rmse"] - f["rmse"])
        qp = np.array(log["qpos"])
        tr = Traj(traj, speed=1.0)
        t = (np.arange(len(qp)) + 1) * DT
        err = np.linalg.norm(qp[:, 21:24] - tr.ref(t)[0], axis=1)   # same definition as run_episode_p2c (P vs ref(t+dt))
        d2 = abs(float(np.sqrt((err ** 2).mean())) - f["rmse"])
        ok = bool(d < 1e-9 and r["success"] == f["success"])
        print(f"{m:10s} {prof:7s} {traj:6s} seed {seed}: rerun {r['rmse']:.12f} formal {f['rmse']:.12f} "
              f"|diff| {d:.2e}  success {r['success']}=={f['success']}  steps {len(qp)}  qpos-rmse diff {d2:.1e}  "
              f"{'MATCH' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            bad.append(e)
        np.savez(cache_path(e), qpos=qp, mjcf=log["mjcf"], rmse=r["rmse"], formal=f["rmse"], diff=d,
                 success=f["success"], err=err)
    if bad:
        print("MISMATCH, no videos will be produced:", bad)
        sys.exit(2)


def _eta(m):
    return "none" if m == "DOB" else "est"


# ---------------------------------------------------------------- render
def render_video(name, spec, D, snap, az=60, el=-16, dist=2.8):
    import mujoco
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from smpc.control.trajectories import Traj
    sides = [m for m in (spec["left"], spec.get("right")) if m]
    Z = [np.load(cache_path((m, spec["prof"], spec["traj"], spec["seed"]))) for m in sides]
    for z, m in zip(Z, sides):   # re-check at render time too
        assert float(z["diff"]) < 1e-9, f"{name}/{m}: rerun RMSE differs from formal"
    mj = str(Z[0]["mjcf"])
    assert all(str(z["mjcf"]) == mj for z in Z), "left/right MJCF differ (different true parameters?)"
    tr = Traj(spec["traj"], speed=1.0)
    ref_path = np.array([tr.ref(t)[0][0] for t in np.linspace(0, 10, 400)])
    snap.sw_context()
    pw = W // len(sides)
    m = mujoco.MjModel.from_xml_string(snap.visual_mjcf(mj, ref_path))
    dat = mujoco.MjData(m)
    scn, opt = mujoco.MjvScene(m, maxgeom=2000), mujoco.MjvOption()
    con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, con)
    vp = mujoco.MjrRect(0, 0, pw, TOP_H)
    rgb = np.zeros((TOP_H, pw, 3), np.uint8)
    cam = mujoco.MjvCamera()

    def shot(z, k):
        dat.qpos[:] = z["qpos"][k]
        mujoco.mj_forward(m, dat)
        cam.lookat[:] = 0.5 * (dat.qpos[21:24] + ref_path.mean(0)) + np.array([0, 0, 0.65])
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        mujoco.mjv_updateScene(m, dat, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(vp, scn, con)
        mujoco.mjr_readPixels(rgb, None, vp, con)
        return np.flipud(rgb).copy()

    nstep = min(len(z["qpos"]) for z in Z)
    ks = list(range(STRIDE - 1, nstep, STRIDE))
    nhold = int(round(HOLD_S * 25))
    errs = [z["err"][:nstep] * 100 for z in Z]
    T = np.arange(1, nstep + 1) * DT
    ymax = max(e.max() for e in errs) * 1.08
    cols = ["#1f6fb4", "#d9611b"][:len(Z)]
    fmt = lambda x: f"{x:.2f}"

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    img_axes = []
    for i in range(len(Z)):
        ax = fig.add_axes([i * pw / W, 1 - TOP_H / H, pw / W, TOP_H / H])
        ax.axis("off")
        img_axes.append(ax.imshow(np.zeros((TOP_H, pw, 3), np.uint8), interpolation="nearest", aspect="auto"))
        ax.set_xlim(-0.5, pw - 0.5)
        ax.set_ylim(TOP_H - 0.5, -0.5)
    axp = fig.add_axes([62 / W, 46 / H, (W - 82) / W, (H - TOP_H - 46 - 30) / H])
    axp.set_xlim(0, T[-1])
    axp.set_ylim(0, ymax)
    axp.set_xlabel("time (s)")
    axp.set_ylabel("|p - p_ref| (cm)")
    axp.grid(alpha=.3)
    lines = []
    for i, e in enumerate(errs):
        axp.plot(T, e, color="#bbbbbb", lw=1.4, zorder=1)
        lines.append(axp.plot([], [], color=cols[i], lw=2.0, zorder=2, label=sides[i])[0])
    axp.legend(loc="upper left", fontsize=9, frameon=False, ncol=len(Z))
    cur = [axp.axvline(0, color="k", lw=0.8, alpha=.5)]
    boxes = []
    for i, mname in enumerate(sides):
        f = get(D, mname, spec["prof"], spec["traj"], spec["seed"])
        txt = (f"{mname}   t = 0.00 s   RMSE {fmt(f['rmse'] * 100)} cm   {'success' if f['success'] else 'FAILED'}")
        boxes.append(fig.text((i * pw + 8) / W, 1 - 14 / H, txt, fontsize=10.5, va="center", ha="left", color="white",
                              family="DejaVu Sans", bbox=dict(boxstyle="round,pad=0.35", fc=cols[i], ec="none", alpha=.92)))
    sub = f"{spec['traj']}  {spec['prof']}  seed {spec['seed']}  (episode RMSE is for the whole episode)"
    fig.text(0.5, 1 - (TOP_H - 10) / H, sub, fontsize=8.5, ha="center", va="center", color="#333333",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=.8))

    OUT.mkdir(parents=True, exist_ok=True)
    mp4 = OUT / f"{name}.mp4"
    ff = subprocess.Popen(["/usr/bin/ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                           "-s", f"{W}x{H}", "-r", "25", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "20", "-preset", "medium", "-movflags", "+faststart", str(mp4)], stdin=subprocess.PIPE)
    first_png = None
    for n_, k in enumerate(ks + [ks[-1]] * nhold):
        for i, z in enumerate(Z):
            img_axes[i].set_data(shot(z, k))
            lines[i].set_data(T[:k + 1], errs[i][:k + 1])
            f = get(D, sides[i], spec["prof"], spec["traj"], spec["seed"])
            boxes[i].set_text(f"{sides[i]}   t = {T[k]:.2f} s   RMSE {fmt(f['rmse'] * 100)} cm   "
                              f"{'success' if f['success'] else 'FAILED'}")
        cur[0].set_xdata([T[k], T[k]])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        assert buf.shape == (H, W, 4), buf.shape
        ff.stdin.write(buf.tobytes())
        if n_ == len(ks) // 2:
            first_png = buf.copy()
    ff.stdin.close()
    assert ff.wait() == 0
    plt.close(fig)
    gif = OUT / f"{name}.gif"
    pal = SCR / f"{name}_pal.png"
    for colors, dith in ((128, "bayer:bayer_scale=5"), (96, "bayer:bayer_scale=5"), (64, "bayer:bayer_scale=5"),
                         (64, "none"), (32, "none")):   # first variant that is < 4.8 MB
        subprocess.run(["/usr/bin/ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                        f"fps=12,scale=720:-1:flags=lanczos,palettegen=max_colors={colors}", str(pal)], check=True)
        subprocess.run(["/usr/bin/ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-i", str(pal), "-lavfi",
                        f"fps=12,scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse=dither={dith}", str(gif)], check=True)
        if gif.stat().st_size < 4.8e6:
            break
    assert gif.stat().st_size < 5e6, "gif too large"
    print(f"{name}: mp4 {mp4.stat().st_size / 1e6:.2f} MB, gif {gif.stat().st_size / 1e6:.2f} MB, "
          f"{len(ks) + nhold} frames", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "record", "render", "all"])
    ap.add_argument("--only", nargs="+")
    a = ap.parse_args()
    D = load_eps()
    S = selection(D)
    if a.only:
        S = {k: v for k, v in S.items() if k in a.only}
    SCR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.only:
        (OUT / "selection.json").write_text(json.dumps(S, indent=1))
    if a.mode == "select":
        print(json.dumps(S, indent=1))
        return
    if a.mode in ("record", "all"):
        record(S, D)
    if a.mode in ("render", "all"):
        snap = _load_snapshot()
        for name, spec in S.items():
            render_video(name, spec, D, snap)


if __name__ == "__main__":
    main()
