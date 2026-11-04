"""Recompute the v3 paper numbers that no other tracked script produces.

Reads only results/prereg/episodes.jsonl (through the frozen loader and the frozen RMSE_eff rule of
experiments/prereg_analyze.py). Single-threaded, runs in a few seconds. Each block prints the value, the
place in paper/workshop/main.tex where it is quoted, and the quoted value with an OK / MISMATCH flag.

  (1) H1a relative effect for s >= 0.5 (7 cells; figure-eight s = 2.5 excluded as a failure-scoring artifact)
      -- abstract and Sec. V-A, "about 11-14%".
  (2) Range of CON-ATT mean tracking error over the 10 cells, speed 1.0, no packet drop
      -- abstract, Introduction, Conclusion, "2.3-3.7 cm".
  (3) Formal DOB - CON-ATT mean at s = 2.5 (RMSE_eff, failures scored 100 cm) -- Fig. 4 caption, "31.8 and 27.2 cm".
  (4) Failed-episode set -- Appendix C: count, per method, non-finite values, range of the last-second mean
      error, largest maximum error; DOB failures per s at speed 1.0.
  (5) Packet drop p = 0.9 versus the lossless run on the same 20 seeds -- Appendix D, Table IV (and Sec. V-B).

Run from the repository root:

    .venv/bin/python paper/workshop/scripts/recompute_v3_numbers.py
"""
from __future__ import annotations

import collections
import importlib.util
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)

_spec = importlib.util.spec_from_file_location("prereg_analyze", os.path.join(ROOT, "experiments", "prereg_analyze.py"))
pa = importlib.util.module_from_spec(_spec)
sys.modules["prereg_analyze"] = pa
_spec.loader.exec_module(pa)

SEEDS = list(range(3000, 3040))
S_LEVELS = [0, 0.5, 1, 1.75, 2.5]
TRAJS = pa.TRAJS
BAD = []


def banner(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


def check(label, ok):
    print(f"    [{'OK' if ok else 'MISMATCH'}] {label}")
    if not ok:
        BAD.append(label)


banner("Load results/prereg/episodes.jsonl")
R = pa.load_rows("results/prereg/episodes.jsonl", set(SEEDS))
ctx = pa.Ctx(R, SEEDS)


def mean_eff(m, s, tr, speed=1.0, p=0.0, seeds=None):
    return float(ctx.vec((m, p), s, speed, tr, seeds).mean())


# ------------------------------------------------------------------ (1)
banner("(1) H1a relative effect for s >= 0.5 (Abstract, Sec. V-A: 'about 11-14%')")
print("rel = mean(EKF-L+GEO - EKF-L) / mean(EKF-L), RMSE_eff, speed 1.0, 40 seeds")
rels = []
for tr in TRAJS:
    for s in S_LEVELS[1:]:
        d = ctx.diff(pa.GEO, pa.EKF, s, 1.0, tr)
        den = mean_eff("EKF-L", s, tr)
        rel = float(d.mean()) / den
        art = (tr == "fig8" and s == 2.5)
        print(f"  {tr:7s} s={s:<5} mean diff {d.mean():+.4f} cm  EKF-L mean {den:.4f} cm  rel {100*rel:+.2f}%"
              + ("   <- failure-scoring artifact, excluded" if art else ""))
        if not art:
            rels.append(-100 * rel)
print(f"  range over the 7 non-artifact cells: {min(rels):.2f}% to {max(rels):.2f}%")
check("paper: about 11-14%  (rounds to 11 and 14)", round(min(rels)) == 11 and round(max(rels)) == 14 and len(rels) == 7)

# ------------------------------------------------------------------ (2)
banner("(2) CON-ATT mean tracking error, speed 1.0, p_drop 0 (Abstract, Intro, Conclusion: '2.3-3.7 cm')")
vals = []
for tr in TRAJS:
    row = []
    for s in S_LEVELS:
        v = mean_eff("CON-ATT", s, tr)
        row.append(v)
        vals.append(v)
    print(f"  {tr:7s} " + "  ".join(f"s={s:<5}{v:.3f}" for s, v in zip(S_LEVELS, row)))
print(f"  range: {min(vals):.3f} to {max(vals):.3f} cm")
check("paper: 2.3-3.7 cm", round(min(vals), 1) == 2.3 and round(max(vals), 1) == 3.7)

# ------------------------------------------------------------------ (3)
banner("(3) Formal DOB - CON-ATT at s = 2.5 (Fig. 4 caption: '31.8 and 27.2 cm')")
f3 = {}
for tr in TRAJS:
    d = ctx.diff(pa.DOB, pa.CON, 2.5, 1.0, tr)
    f3[tr] = float(d.mean())
    print(f"  {tr:7s} mean (DOB - CON-ATT, RMSE_eff) = {d.mean():.3f} cm")
check("paper: 31.8 (circle) and 27.2 (figure-eight)", round(f3["circle"], 1) == 31.8 and round(f3["fig8"], 1) == 27.2)

# ------------------------------------------------------------------ (4)
banner("(4) Failed-episode set (Appendix C)")
allrows = [r for d in R.values() for r in d.values()]
fails = [r for r in allrows if not r.get("success", False)]
print(f"  episodes loaded: {len(allrows)}; failed (success False): {len(fails)}")
bym = collections.Counter(r["method"] for r in fails)
print("  per method: " + ", ".join(f"{m} {n}" for m, n in sorted(bym.items(), key=lambda kv: -kv[1])))
nonfinite = [r for r in fails if (not r.get("finite", True)) or not math.isfinite(float(r["rmse"]))
             or not math.isfinite(float(r["final_err"])) or not math.isfinite(float(r["max_err"]))]
print(f"  failed episodes with a non-finite value: {len(nonfinite)}")
nonfinite_all = [r for r in allrows if (not r.get("finite", True))]
print(f"  episodes with finite=False among all {len(allrows)}: {len(nonfinite_all)}")
fe = np.array([100 * float(r["final_err"]) for r in fails])
me = np.array([100 * float(r["max_err"]) for r in fails])
print(f"  last-second mean error of failed episodes: {fe.min():.2f} to {fe.max():.2f} cm "
      f"(success threshold 10 cm; all above: {bool((fe > 10).all())})")
print(f"  largest maximum error among failed episodes: {me.max():.2f} cm")
print("  DOB failures per s (speed 1.0, p_drop 0, both trajectories, 80 episodes per s):")
dob = collections.Counter()
for (m, prof, speed, p, tr), d in R.items():
    if m == "DOB" and speed == 1.0 and p == 0.0:
        dob[prof] += sum(1 for r in d.values() if not r.get("success", False))
for s in S_LEVELS:
    print(f"    s={s:<5} {dob.get(f's={s:g}', 0)} / 80")
dob_other = sum(1 for r in fails if r["method"] == "DOB" and float(r["speed"]) != 1.0)
print(f"  DOB failures at other speeds: {dob_other}")
check("paper: 35 failed (DOB 30, LOC-M 4, EKF-L 1)", len(fails) == 35 and bym == {"DOB": 30, "LOC-M": 4, "EKF-L": 1})
check("paper: none non-finite", len(nonfinite) == 0)
check("paper: last-second mean error 10.1 to 20.3 cm", round(fe.min(), 1) == 10.1 and round(fe.max(), 1) == 20.3)
check("paper: no maximum error above 23 cm", bool(me.max() <= 23.0))
check("paper: DOB misses the criterion in 22 of 80 at s = 2.5", dob.get("s=2.5", 0) == 22)

# ------------------------------------------------------------------ (5)
banner("(5) Packet drop p = 0.9 vs lossless, same seeds (Appendix D, Table IV)")
s9 = sorted(set(R[("CON-ATT", "s=1", 1.0, 0.9, "circle")]) & set(R[("CON-ATT", "s=1", 1.0, 0.9, "fig8")]))
print(f"  p=0.9 seeds: {s9[0]}..{s9[-1]} (n={len(s9)}); the lossless run is restricted to the same seeds")
FIELDS = [("tracking error cm (RMSE_eff)", None), ("zeta_hat RMS, 2nd half (zeta_rms_h2)", "zeta_rms_h2"),
          ("dm error at t=5 s (dm_err@5)", "dm_err@5"), ("0.5 s-ahead tension error N (tpe10)", "tpe10")]
quoted = {  # paper Table IV: (lossless circle, fig8), (p=0.9 circle, fig8), change (circle, fig8), in percent
    None: ((2.88, 2.91), (2.80, 2.90), None),
    "zeta_rms_h2": ((0.0138, 0.0140), (0.0195, 0.0194), (41, 39)),
    "dm_err@5": ((0.0446, 0.0476), (0.0529, 0.0542), (19, 14)),
    "tpe10": ((0.238, 0.241), (0.292, 0.292), (23, 22)),
}
dig = {None: 2, "zeta_rms_h2": 4, "dm_err@5": 4, "tpe10": 3}
for name, fld in FIELDS:
    l0, l9, ch = [], [], []
    for tr in TRAJS:
        if fld is None:
            a = ctx.vec(("CON-ATT", 0.0), 1, 1.0, tr, s9).mean()
            b = ctx.vec(("CON-ATT", 0.9), 1, 1.0, tr, s9).mean()
        else:
            a = np.mean([R[("CON-ATT", "s=1", 1.0, 0.0, tr)][sd][fld] for sd in s9])
            b = np.mean([R[("CON-ATT", "s=1", 1.0, 0.9, tr)][sd][fld] for sd in s9])
        l0.append(float(a)); l9.append(float(b)); ch.append(100 * (float(b) / float(a) - 1))
    print(f"  {name}")
    print(f"    lossless {l0[0]:.5f} / {l0[1]:.5f}   p=0.9 {l9[0]:.5f} / {l9[1]:.5f}   change {ch[0]:+.1f}% / {ch[1]:+.1f}%")
    q0, q9, qc = quoted[fld]
    d = dig[fld]
    ok = (tuple(round(x, d) for x in l0) == q0 and tuple(round(x, d) for x in l9) == q9
          and (qc is None or tuple(round(x) for x in ch) == qc))
    check(f"paper Table IV row '{name}'", ok)

banner("(5b) p = 0.9 vs EKF-L+GEO on the same seeds (Appendix D text: -0.25 and -0.21 cm)")
q = {"circle": -0.25, "fig8": -0.21}
for tr in TRAJS:
    x = ctx.vec(("CON-ATT", 0.9), 1, 1.0, tr, s9) - ctx.vec(("EKF-L+GEO", 0.0), 1, 1.0, tr, s9)
    lo, hi = pa.boot_ci(x, 95)
    print(f"  {tr:7s} mean {x.mean():+.4f} cm  95% CI [{lo:+.3f}, {hi:+.3f}]")
    check(f"paper {tr}: {q[tr]:+.2f}", round(float(x.mean()), 2) == q[tr])

banner("Summary")
print("All quoted values match." if not BAD else "MISMATCHES: " + "; ".join(BAD))
