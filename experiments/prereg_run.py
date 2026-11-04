"""PREREG 正式實驗執行腳本（只依 docs/PREREG.md §2、§3.2、§9、§10.1 寫成）。

用法：
  python experiments/prereg_run.py --dry     # 只列工作數與前 5 個，不跑模擬
  python experiments/prereg_run.py           # 真的跑（seeds 3000-3039），結果 append 到 results/prereg/episodes.jsonl，可續跑

注意：不用 p2c_common.run_jobs（其 _work 有 assert seed < 3000 且不傳 n），自帶相同邏輯的執行器；見 PREREG §10.1。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

OUT = "results/prereg/episodes.jsonl"
SEEDS = list(range(3000, 3040))
TRAJS = ["circle", "fig8"]
S_LIST = [0, 0.5, 1, 1.75, 2.5]
G1_METHODS = ["DOB", "LOC-M", "EKF-L", "EKF-L+GEO", "CEN-ATT", "CON-ATT"]

# §3.2 凍結 hp（出處 results/p2d/best.json；HYB 出處 results/p2c/best.json）
HP = {
    "DOB": {"fc": 16.0},
    "LOC-M": {"lam": 0.998, "prior": 1.0},
    "EKF-L": {"q_b": 5e-4, "r_scale": 10.0},
    "EKF-L+GEO": {"q_b": 5e-4, "r_scale": 10.0, "use_dc": 0},
    "CEN-ATT": {"lam": 0.9998, "prior": 2.0},
    "CON-ATT": {"lam": 0.9998, "prior": 2.0, "iters": 2, "p_drop": 0.0},
}
HP_HYB = {1.0: {"fc": 8.0, "tau": 0.07}, 1.5: {"fc": 1000.0, "tau": 0.07}}
EST = {"LOC-M", "EKF-L", "CEN-ATT", "CON-ATT", "EKF-L+GEO", "HYB"}   # DOB: none；ORACLE: oracle


def eta_of(m):
    return "oracle" if m == "ORACLE" else ("est" if m in EST else "none")


def hk(hp):
    return ",".join(f"{k}={v:g}" for k, v in sorted(hp.items()))


def mk(m, s, tr, speed, seed, hp):
    prof = f"s={s:g}"
    return dict(key=f"prereg|{m}|{prof}|{tr}|{speed}|{seed}|{hk(hp)}", tag="prereg", method=m, profile=prof,
                speed=speed, traj=tr, seed=seed, hp=hp, eta_comp=eta_of(m), n=3)


def build_jobs():
    J = []
    # G1：6 方法 × 5 s × 2 軌跡 × 40 seeds，speed 1.0
    for m in G1_METHODS:
        for s in S_LIST:
            for tr in TRAJS:
                J += [mk(m, s, tr, 1.0, sd, dict(HP[m])) for sd in SEEDS]
    # G2：6 方法 × s=1 × speed 1.5（同一組 hp）
    for m in G1_METHODS:
        for tr in TRAJS:
            J += [mk(m, 1, tr, 1.5, sd, dict(HP[m])) for sd in SEEDS]
    # G4：CON-ATT p_drop=0.5（40 seeds）、0.9（前 20 seeds），s=1 speed 1.0
    for p, seeds in ((0.5, SEEDS), (0.9, SEEDS[:20])):
        for tr in TRAJS:
            hp = dict(HP["CON-ATT"], p_drop=p)
            J += [mk("CON-ATT", 1, tr, 1.0, sd, hp) for sd in seeds]
    # G5：HYB（speed 1.0、1.5）、ORACLE（speed 1.0），s=1
    for sp in (1.0, 1.5):
        for tr in TRAJS:
            J += [mk("HYB", 1, tr, sp, sd, dict(HP_HYB[sp])) for sd in SEEDS]
    for tr in TRAJS:
        J += [mk("ORACLE", 1, tr, 1.0, sd, {}) for sd in SEEDS]
    return J


def group_of(j):
    if j["method"] in ("HYB", "ORACLE"):
        return "G5"
    if j["hp"].get("p_drop", 0) > 0:
        return "G4"
    return "G1" if j["speed"] == 1.0 else "G2"


def _work(j):
    import torch
    torch.set_num_threads(1)
    from smpc.eval.episode_p2c import run_episode_p2c
    t0 = time.time()
    r = run_episode_p2c(j["method"], j["traj"], j["seed"], j["profile"], j["speed"], j["hp"] or None,
                        eta_comp=j["eta_comp"], n=j["n"])
    r.update(key=j["key"], tag=j["tag"], wall_s=time.time() - t0)
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
    dump = lambda o: o.tolist() if hasattr(o, "tolist") else str(o)
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
                fh.write(json.dumps(r, default=dump) + "\n")
                fh.flush()
                n_done += 1
                if n_done % 10 == 0:
                    print(f"{n_done}/{len(J)} {time.time()-t0:.0f}s W={W} {r['key']} rmse={r['rmse']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只列工作數量與前 5 個工作，不跑")
    a = ap.parse_args()
    jobs = build_jobs()
    keys = [j["key"] for j in jobs]
    assert len(set(keys)) == len(keys), "key 重複"
    if a.dry:
        print(f"工作總數 {len(jobs)}（PREREG §9：3240）", "OK" if len(jobs) == 3240 else "不符！")
        for g, c in sorted(Counter(group_of(j) for j in jobs).items()):
            print(f"  {g}: {c}")
        for j in jobs[:5]:
            print(" ", json.dumps(j))
        return
    run_jobs(jobs, OUT)


if __name__ == "__main__":
    sys.exit(main())
