"""P2c 共用：可續跑的 jobs 執行器（每集一行寫進 jsonl）。
平行度依 available 記憶體動態調整（每個 worker 約 190 MB 增量；available 偏低時降到 1 個，最多 3 個）。
seeds：調參 2020-2029（實際用 2020-2024）、評估 2100-2119；3000 以上禁用。
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

EVAL_SEEDS = list(range(2100, 2120))
TUNE_SEEDS = list(range(2020, 2025))
PROFILES = ["clean", "moderate", "harsh"]
SPEEDS = [1.0, 1.5]
TRAJS = ["circle", "fig8"]


def eta_policy(method, profile):
    """事先決定（跑之前）：有推力係數誤差的 profile 中，CEN-ATT 用自己的 ζ̂ 補償（ζ 是它的本地參數），ORACLE 用真值；
    其他方法不補償（DOB/HYB 的殘差本來就吸收推力係數誤差）。clean 一律不補償（同 P2b）。"""
    if profile == "clean":
        return "none"
    return {"CEN-ATT": "est", "ORACLE": "oracle"}.get(method, "none")


def _work(j):
    import torch
    torch.set_num_threads(1)
    from smpc.eval.episode_p2c import run_episode_p2c
    assert j["seed"] < 3000, "3000 以上保留"
    t0 = time.time()
    r = run_episode_p2c(j["method"], j["traj"], j["seed"], j["profile"], j["speed"], j.get("hp"),
                        eta_comp=j.get("eta_comp", "none"), oracle_acc=j.get("oracle_acc", False))
    r.update(key=j["key"], tag=j.get("tag", ""), wall_s=time.time() - t0)
    return r


def avail_mb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return int(l.split()[1]) / 1024
    return 0


def run_jobs(jobs, path, max_workers=3):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["key"] for l in open(path) if l.strip()}
    J = [j for j in jobs if j["key"] not in done]
    print(f"待跑 {len(J)} 集（已完成 {len(done)}）", flush=True)
    t0, n_done, it = time.time(), 0, iter(J)
    inflight, exhausted = set(), False
    with ProcessPoolExecutor(max_workers) as ex, open(path, "a") as fh:
        while inflight or not exhausted:
            W = int(np.clip((avail_mb() - 8200) // 190 + len(inflight), 1, max_workers))
            while not exhausted and len(inflight) < W:
                try:
                    inflight.add(ex.submit(_work, next(it)))
                except StopIteration:
                    exhausted = True
            if not inflight:
                break
            dn, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for f in dn:
                r = f.result()
                fh.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
                fh.flush()
                n_done += 1
                if n_done % 10 == 0:
                    print(f"{n_done}/{len(J)} {time.time()-t0:.0f}s W={W} avail={avail_mb():.0f}MB {r['key']} rmse={r['rmse']:.3f}", flush=True)


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def boot_ci(x, B=4000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    mb = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(mb, 2.5)), float(np.percentile(mb, 97.5))


def boot_p(x, B=20000, seed=1):
    """雙尾 bootstrap p 值（以 0 為界）；下限 1/B。"""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    mb = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    p = 2 * min((mb <= 0).mean(), (mb >= 0).mean())
    return float(max(p, 1.0 / B))


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj
