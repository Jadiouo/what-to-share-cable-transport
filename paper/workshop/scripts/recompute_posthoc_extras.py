"""Post-hoc extras for the workshop paper appendix (three independent blocks).

  (a) ORACLE reference: mean RMSE_eff (cm) at s=1, speed 1.0, no packet drop, 40 seeds, plus
      EKF-L - ORACLE, EKF-L - CON-ATT and their ratio; cross-checked against results/prereg/cells.csv.
  (b) CON-ATT message-layer completeness: `pose_ok_frac` (fraction of (step, agent) pairs for which the
      agent holds a fresh packet from every team member) under p_drop in {0, 0.5, 0.9}, driven by synthetic
      inputs (no simulation), seeds 3000-3039, 1000 steps each.
  (c) Two-sided sign-flip p-values for the 20 F1 tests (H1a, H1b), with Holm adjustment. The one-sided p and
      Holm values are first recomputed with the frozen analysis functions and compared with
      results/prereg/tests.csv; the script stops if they disagree.
  Added in the fifth batch (blocks (d)-(f), printed after (c); (a)-(c) are unchanged):
  (d) H4 (F5) slope with raw-RMSE scoring (sensitivity-(a) rule): same test as the preregistration (shared sign
      matrix, per-seed OLS slope, one-sided), formal scoring first checked against tests.csv.
  (e) Message sizes per control step for EKF-L+GEO and CON-ATT, read from the real classes' constants.
  (f) clip_frac (parameter-estimate clipping fraction, not thrust saturation) for CON-ATT at p_drop 0.9 vs
      lossless on seeds 3000-3019, and p_drop 0.5 vs lossless on 3000-3039; checked against cells.csv.

Run from the repository root (single-threaded, well under 60 s):

    .venv/bin/python paper/workshop/scripts/recompute_posthoc_extras.py

Everything statistical is imported from experiments/prereg_analyze.py (rmse_eff, load_rows, sign_matrix,
signflip_p, holm, TIE_TOL, ...); nothing is re-implemented here. Block (b) calls the real
smpc.adapt.con_att.ConATT message layer.
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

_spec = importlib.util.spec_from_file_location("prereg_analyze", os.path.join(ROOT, "experiments", "prereg_analyze.py"))
pa = importlib.util.module_from_spec(_spec)
sys.modules["prereg_analyze"] = pa
_spec.loader.exec_module(pa)

SEEDS = list(range(3000, 3040))
T0 = time.time()
FAILED = []


def banner(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


# ---------------------------------------------------------------------------- load frozen results
banner("Load results/prereg/episodes.jsonl (frozen loader; its own log line follows)")
R = pa.load_rows("results/prereg/episodes.jsonl", set(SEEDS))
ctx = pa.Ctx(R, SEEDS)

# ---------------------------------------------------------------------------- (a)
banner("(a) ORACLE reference, s=1, speed 1.0, p_drop=0, 40 seeds (RMSE_eff in cm, rmse_eff rule)")
cells = {}
with open("results/prereg/cells.csv") as f:
    for r in csv.DictReader(f):
        cells[(r["method"], r["profile"], float(r["speed"]), float(r["p_drop"]), r["traj"])] = r

METHODS_A = ["ORACLE", "EKF-L", "EKF-L+GEO", "CON-ATT"]
mean_a = {}
n_a = {}
for tr in pa.TRAJS:
    for m in METHODS_A:
        v = ctx.vec((m, 0), 1, 1.0, tr)          # key uses p_drop=0 -> rows with no p_drop key or p_drop 0
        mean_a[(m, tr)] = float(v.mean())
        n_a[(m, tr)] = len(v)
print(f"{'method':<11}{'traj':<8}{'n':>3}{'mean RMSE_eff cm':>18}{'cells.csv':>12}{'|diff|':>11}  check")
for tr in pa.TRAJS:
    for m in METHODS_A:
        c = float(cells[(m, "s=1", 1.0, 0.0, tr)]["rmse_eff_mean_cm"])
        d = abs(mean_a[(m, tr)] - c)
        ok = d < 5e-5 and int(cells[(m, "s=1", 1.0, 0.0, tr)]["n"]) == n_a[(m, tr)] == 40
        if not ok:
            FAILED.append(f"(a) cells.csv mismatch {m} {tr}")
        print(f"{m:<11}{tr:<8}{n_a[(m, tr)]:>3}{mean_a[(m, tr)]:>18.5f}{c:>12.5f}{d:>11.2e}  {'OK' if ok else 'MISMATCH'}")
print()
print(f"{'traj':<8}{'EKF-L - ORACLE':>16}{'EKF-L - CON-ATT':>17}{'(EKF-L - CON-ATT)/(EKF-L - ORACLE)':>38}")
for tr in pa.TRAJS:
    gap = mean_a[("EKF-L", tr)] - mean_a[("ORACLE", tr)]
    gain = mean_a[("EKF-L", tr)] - mean_a[("CON-ATT", tr)]
    print(f"{tr:<8}{gap:>16.5f}{gain:>17.5f}{gain / gap:>38.5f}")
print("(CON-ATT closes the fraction in the last column of the EKF-L-to-ORACLE gap; all means in cm.)")
if FAILED:
    print("\nSTOP: cross-check failed:", FAILED)
    sys.exit(1)

# ---------------------------------------------------------------------------- (b)
banner("(b) CON-ATT message layer: pose_ok_frac under packet drop (synthetic drive, seeds 3000-3039, 1000 steps)")
print("""What pose_ok_frac measures (from src/smpc/adapt/con_att.py):
  Each step, each agent i checks its table of last-received measurement packets (one per team member, own
  included): the packet from every source o must satisfy age = k - stamp[i,o] <= max_age (default 5 steps = 50 ms).
  pose_ok_frac[k] = mean over the 3 agents of that boolean, i.e. the fraction of agents that used the full-team
  pose-aware rows at step k. Packets are forwarded over relay_rounds = N-1 = 2 rounds per step on the path graph
  1-2-3, plus `iters`=2 consensus rounds; p_drop acts independently on every directed message (j->i) in every
  relay and consensus round (rng = default_rng([seed, 6]), _recv). A stale packet is reused until it exceeds 5
  steps. The script reports mean over steps, then mean/min/max across seeds.
  Synthetic-vs-flight (inference from the code, checked empirically below): the ok flag depends only on the
  integer stamps, which are written from the step counter and the drop mask; the mask comes from an rng seeded by
  (seed, 6) and consumed in a fixed order (relay rounds, then consensus rounds) independent of the packet
  contents, and sensing delay (SensorModel.delay_bundle) only delays packet contents, not stamps. So the fraction
  does not depend on flight state, and the synthetic drive with the same seed should reproduce the in-flight value
  exactly. Caveat: an in-flight episode that stops early (non-finite state) would have fewer steps.""")
import torch  # noqa: E402  (imported only to pin threads; the estimator itself is numpy)

torch.set_num_threads(1)
from smpc.adapt.con_att import ConATT  # noqa: E402
from smpc.adapt.estimators import Geometry  # noqa: E402
from smpc.eval.episode import make_env  # noqa: E402

STEPS = 1000
env = make_env(3, 1.0, (0, 0, 0))
geom = Geometry.from_env(env, 30.0, 1.0, 1.0)
pq = env.hover_positions(np.zeros(3))
q = env.attach_body - pq
q /= np.linalg.norm(q, axis=1, keepdims=True)
a0, f_, y_, act = np.zeros(3), np.ones((3, 3)), np.ones((3, 3)) * 0.1, np.ones(3, bool)


class ConATTCommsOnly(ConATT):
    """Same message layer (stamps, relay, drop masks, consensus rounds, rng order); the estimation maths in
    _local_info is replaced by the freshness test only, so 40 seeds x 1000 steps fit in the time budget.
    Its pose_ok_frac is verified below against the full ConATT on a subset of seeds."""

    def _local_info(self, i, a_ref, x):
        ok = bool(np.all(self.k - self.stamp[i] <= self.max_age))
        return np.zeros((self.d, self.d)), np.zeros(self.d), ok


def drive(p, seed, steps=STEPS, full=False):
    cls = ConATT if full else ConATTCommsOnly
    con = cls(geom, env.L, sigma_f=0.6, sigma_q=0.01, p_drop=p, seed=seed)
    for _ in range(steps):
        con.update(a0, q, f_, y_, act, pq)
    return float(np.mean(con.pose_ok_frac)), len(con.pose_ok_frac)


t_b = time.time()
B_SEEDS, B_STEPS = SEEDS, STEPS
print("\n[scale] full: 40 seeds x 1000 steps x 3 drop rates, using the comms-only subclass of ConATT (see class docstring).")
print("[verify] comms-only vs full ConATT (real estimator maths), 1000 steps, per-episode mean pose_ok_frac:")
for p, sd in ((0.5, 3000), (0.5, 3017), (0.9, 3000), (0.9, 3039)):
    vf, _ = drive(p, sd, full=True)
    vc, _ = drive(p, sd)
    print(f"    p_drop={p} seed={sd}: full={vf:.6f} comms-only={vc:.6f} identical={vf == vc}")
    if vf != vc:
        FAILED.append(f"(b) comms-only differs from full ConATT at p={p} seed={sd}")

print(f"\n{'p_drop':>7}{'n_seeds':>9}{'steps':>7}{'mean':>10}{'min':>10}{'max':>10}")
okfrac = {}
for p in (0.0, 0.5, 0.9):
    vals = []
    for sd in B_SEEDS:
        v, n_steps = drive(p, sd)
        assert n_steps == B_STEPS
        vals.append(v)
    vals = np.array(vals)
    okfrac[p] = vals
    print(f"{p:>7.1f}{len(vals):>9}{B_STEPS:>7}{vals.mean():>10.4f}{vals.min():>10.4f}{vals.max():>10.4f}")
print("(mean = average over steps within a seed, then over seeds; min/max are across seeds. p_drop=0 gives exactly 1:"
      " relay delivers every packet within the step.)")

# empirical check: the ok flag does not depend on packet contents (flight state)
rng = np.random.default_rng(1)
sd = B_SEEDS[0]
a = ConATT(geom, env.L, sigma_f=0.6, sigma_q=0.01, p_drop=0.5, seed=sd)
b = ConATT(geom, env.L, sigma_f=0.6, sigma_q=0.01, p_drop=0.5, seed=sd)
for _ in range(300):
    a.update(a0, q, f_, y_, act, pq)
    qn = q + 0.05 * rng.standard_normal(q.shape)
    qn /= np.linalg.norm(qn, axis=1, keepdims=True)
    b.update(a0 + rng.standard_normal(3), qn, f_ * (1 + rng.random((3, 3))), y_ + rng.standard_normal((3, 3)),
             rng.random(3) > 0.2, pq + 0.1 * rng.standard_normal((3, 3)))
same = np.array_equal(a.pose_ok_frac, b.pose_ok_frac)
print(f"\n[check] same seed, 300 steps, p_drop=0.5, different packet contents/active flags: "
      f"pose_ok_frac sequences identical = {same}")
if not same:
    FAILED.append("(b) pose_ok_frac depends on packet contents")
print(f"[block (b) time] {time.time() - t_b:.1f} s")

# ---------------------------------------------------------------------------- (c)
banner("(c) F1 family: recompute one-sided p / Holm, compare with tests.csv, then two-sided p / Holm")
seeds_t = list(SEEDS)
specs = [s for s in pa.build_specs(seeds_t) if s["family"] == "F1"]
res, skipped = pa.run_tests(ctx, specs, allow_partial=False, families=["F1"])
assert len(res) == 20 and not skipped

tests = {}
with open("results/prereg/tests.csv") as f:
    for r in csv.DictReader(f):
        if r["family"] == "F1":
            tests[r["test_id"]] = r
print(f"{'test_id':<26}{'p_one(new)':>12}{'p_one(csv)':>12}{'holm(new)':>12}{'holm(csv)':>12}  check")
bad = 0
for r in res:
    t = tests[pa.tid(r["sp"])]
    d1 = abs(r["p"] - float(t["p_onesided"]))
    d2 = abs(r["p_adj"] - float(t["p_holm_adj"]))
    ok = d1 < 5e-6 and d2 < 5e-6 and (str(r["rej"]) == t["rejected"])
    bad += not ok
    print(f"{pa.tid(r['sp']):<26}{r['p']:>12.6f}{float(t['p_onesided']):>12.6f}{r['p_adj']:>12.6f}{float(t['p_holm_adj']):>12.6f}  "
          f"{'OK' if ok else 'MISMATCH'}")
if bad:
    FAILED.append(f"(c) {bad} F1 rows disagree with tests.csv")
    print(f"\nSTOP: {bad} rows disagree with tests.csv; two-sided p not computed.")
    print("\nFAILED CHECKS:", FAILED)
    sys.exit(1)
print("All 20 rows agree with tests.csv to < 5e-6 (rejection flags identical).")

# two-sided p with the same sign matrix and tolerance as the frozen one-sided p
S = pa.sign_matrix(len(seeds_t))
for r in res:
    x = np.asarray(r["x"], float)
    mb = S @ x / len(x)
    m = x.mean()
    tol = pa.TIE_TOL * max(1.0, float(np.abs(x).max()))
    r["p2"] = float((1 + np.sum(np.abs(mb) >= abs(m) - tol)) / (S.shape[0] + 1))
adj2, rej2 = pa.holm([r["p2"] for r in res])
for r, a_, j in zip(res, adj2, rej2):
    r["p2_adj"], r["rej2"] = float(a_), bool(j)

print(f"\nB = {S.shape[0]} sign flips (shared matrix); alpha = {pa.ALPHA}; Holm over the 20 F1 tests, one-sided and two-sided separately.")
print("Direction: mean_diff_cm < 0 means the first-named method (H1a: EKF-L+GEO vs EKF-L; H1b: CON-ATT vs EKF-L+GEO) is better.\n")
print(f"{'hyp':<5}{'s':>5}  {'traj':<7}{'mean_diff_cm':>13}{'Holm p 1-sided':>16}{'Holm p 2-sided':>16}{'rej 1s':>8}{'rej 2s':>8}")
rows = sorted(res, key=lambda r: (r["sp"]["hyp"], r["sp"]["s"], r["sp"]["traj"]))
for r in rows:
    sp = r["sp"]
    print(f"{sp['hyp']:<5}{sp['s']:>5g}  {sp['traj']:<7}{r['mean']:>13.4f}{r['p_adj']:>16.6f}{r['p2_adj']:>16.6f}"
          f"{str(r['rej']):>8}{str(r['rej2']):>8}")

print("\n(i) Cells rejected one-sided but not two-sided:")
lst = [r for r in rows if r["rej"] and not r["rej2"]]
if not lst:
    print("    none")
for r in lst:
    sp = r["sp"]
    print(f"    {sp['hyp']} s={sp['s']:g} {sp['traj']}: mean_diff {r['mean']:.4f} cm, Holm p one-sided {r['p_adj']:.6f}, two-sided {r['p2_adj']:.6f}")

print("\n(ii) (hyp, s) 'supported' = rejected on BOTH trajectories; supported one-sided but not two-sided:")
lst2, n1, n2 = [], 0, 0
for h in ("H1a", "H1b"):
    for s in pa.S_LIST:
        by = {r["sp"]["traj"]: r for r in res if r["sp"]["hyp"] == h and r["sp"]["s"] == s}
        s1 = all(by[t]["rej"] for t in pa.TRAJS)
        s2 = all(by[t]["rej2"] for t in pa.TRAJS)
        n1 += s1
        n2 += s2
        print(f"    {h} s={s:<5g} one-sided supported={s1!s:<6} two-sided supported={s2!s:<6}")
        if s1 and not s2:
            lst2.append((h, s))
print(f"    supported one-sided: {n1}/10; two-sided: {n2}/10")
print("    supported one-sided but not two-sided:", lst2 if lst2 else "none")

# ---------------------------------------------------------------------------- (d)
banner("(d) H4 (F5) with raw-RMSE scoring (sensitivity-(a) rule, p2d_rule=True), same test as the preregistration")
print("""Same functions as the frozen analysis: Ctx.slope_Y / ols_slope (per-seed OLS slope in s of Da-Db), the shared
sign matrix, signflip_p (one-sided, H0: mean slope >= 0), Holm within the F5 family (2 tests: circle, fig8).
Only the scoring changes: p2d_rule=True scores a failed episode (success=False, finite, RMSE <= 1 m) by its raw
RMSE; non-finite or > 1 m episodes stay at 100 cm. (This is the rule of sensitivity (a) in prereg_analyze.py.)
Holm here is within F5 only (2 tests, as in the formal run); the unadjusted one-sided p is the headline number.""")
f5 = [s_ for s_ in pa.build_specs(list(SEEDS)) if s_["family"] == "F5"]
res_f, _ = pa.run_tests(ctx, f5, allow_partial=False, families=["F5"])
res_r, _ = pa.run_tests(ctx, f5, allow_partial=False, p2d=True, families=["F5"])
t5 = {}
with open("results/prereg/tests.csv") as f:
    for r in csv.DictReader(f):
        if r["family"] == "F5":
            t5[r["test_id"]] = r
print(f"{'test':<28}{'slope(new)':>11}{'slope(csv)':>11}{'p1(new)':>11}{'p1(csv)':>11}{'holm(new)':>11}{'holm(csv)':>11}  check")
for r in res_f:
    t = t5[pa.tid(r["sp"])]
    ok = (abs(r["mean"] - float(t["mean_diff_cm"])) < 5e-6 and abs(r["p"] - float(t["p_onesided"])) < 5e-6
          and abs(r["p_adj"] - float(t["p_holm_adj"])) < 5e-6)
    if not ok:
        FAILED.append(f"(d) F5 formal does not reproduce tests.csv: {pa.tid(r['sp'])}")
    print(f"{pa.tid(r['sp']):<28}{r['mean']:>11.4f}{float(t['mean_diff_cm']):>11.4f}{r['p']:>11.6f}{float(t['p_onesided']):>11.6f}"
          f"{r['p_adj']:>11.6f}{float(t['p_holm_adj']):>11.6f}  {'OK' if ok else 'MISMATCH'}")
print("\nRaw-RMSE scoring (post hoc):")
print(f"{'traj':<8}{'slope cm/unit s':>16}{'95% CI':>20}{'p one-sided':>13}{'Holm in F5':>12}")
for r in res_r:
    print(f"{r['sp']['traj']:<8}{r['mean']:>16.4f}{'[%.3f, %.3f]' % r['ci95']:>20}{r['p']:>13.6f}{r['p_adj']:>12.6f}")
print("\nSlope without the s=0 end point (per-seed OLS over s>=0.5; one-sided p not part of any claim, shown for reference):")
print(f"{'traj':<8}{'scoring':<10}{'slope all s':>12}{'slope s>=0.5':>14}{'p one-sided (s>=0.5)':>22}")
for lab, rs in (("formal", res_f), ("raw", res_r)):
    for r in rs:
        Y = r["Y"]
        sl = pa.ols_slope(Y[:, 1:], pa.S_LIST[1:])
        print(f"{r['sp']['traj']:<8}{lab:<10}{r['mean']:>12.4f}{sl.mean():>14.4f}{pa.signflip_p(sl):>22.6f}")
print("(Compare: recompute_h4_sens.py prints circle formal s>=0.5 -0.16; fig8 raw all-s -0.28 and s>=0.5 -0.096.)")

# ---------------------------------------------------------------------------- (e)
banner("(e) Message sizes per control step (counts of numbers, from code constants; not measured traffic)")
from smpc.adapt.ekf_geo import EKFLGeo  # noqa: E402

N = 3
geo_est = EKFLGeo(geom, env.L, 0.05, 0.15)
con_est = ConATT(geom, env.L, sigma_f=0.15, sigma_q=0.01, iters=2)
pk_geo, pk_con = geo_est.data.shape[-1], con_est.data.shape[-1]
d = con_est.d
tri = d * (d + 1) // 2
print(f"EKF-L+GEO: packet per source = {pk_geo} numbers (ekf_geo.py data shape {geo_est.data.shape}: pq(3)+q(3)); "
      f"relay rounds per step = {geo_est.relay_rounds} (= N-1); consensus iterations = none; no estimate exchanged.")
print(f"CON-ATT : packet per source = {pk_con} numbers (con_att.py data shape {con_est.data.shape}: pq3+q3+y3+f3+active1); "
      f"relay rounds per step = {con_est.relay_rounds}; consensus iterations per step = {con_est.iters}.")
print(f"CON-ATT : full-team state dimension d = {d}; one information pair (Lambda, xi) = {tri} (upper triangle of the "
      f"symmetric {d}x{d} Lambda) + {d} (xi) = {tri + d} numbers; = {d * d} + {d} = {d * d + d} if the full matrix is counted.")
print(f"CON-ATT : each consensus message carries two pairs (prior pair and new-information pair): "
      f"{2 * (tri + d)} numbers (upper triangle) or {2 * (d * d + d)} (full matrices); per step x {con_est.iters} iterations = "
      f"{con_est.iters * 2 * (tri + d)} (upper) / {con_est.iters * 2 * (d * d + d)} (full) numbers per directed link.")
print(f"A table of all N={N} source packets: EKF-L+GEO {N * pk_geo} numbers, CON-ATT {N * pk_con} numbers.")
# symmetry check on a short run with non-trivial inputs, so that 'upper triangle' is a fair count
rng_e = np.random.default_rng(5)
con_chk = ConATT(geom, env.L, sigma_f=0.6, sigma_q=0.01, iters=2, seed=1)
for _ in range(20):
    qn = q + 0.05 * rng_e.standard_normal(q.shape)
    qn /= np.linalg.norm(qn, axis=1, keepdims=True)
    con_chk.update(a0 + rng_e.standard_normal(3), qn, f_ * (1 + rng_e.random((3, 3))), y_ + rng_e.standard_normal((3, 3)),
                   np.ones(3, bool), pq + 0.1 * rng_e.standard_normal((3, 3)))
asym = float(np.abs(con_chk.Lam - np.swapaxes(con_chk.Lam, 1, 2)).max())
print(f"[check] max |Lambda - Lambda^T| after 20 steps with random inputs = {asym:.2e} (symmetric -> upper-triangle count valid)")
if (pk_geo, pk_con, d, con_est.relay_rounds, con_est.iters) != (6, 13, 6, 2, 2) or asym > 1e-9:
    FAILED.append("(e) message-size constants differ from the values stated in the paper")

# ---------------------------------------------------------------------------- (f)
banner("(f) clip_frac under packet drop (CON-ATT, s=1, speed 1.0): lossless vs p=0.9 (20 seeds), vs p=0.5 (40 seeds)")
print("""clip_frac (src/smpc/eval/episode_p2c.py lines 154-155, 198): for each control step and each quadrotor i the
parameter estimate theta_i = (dm, dc_x, dc_y) is clipped to the box TH_CLIP = ([-0.7,-0.15,-0.15], [1.0,0.15,0.15])
(episode_est.py line 26) before it reaches the controller; clipn counts (step, agent) pairs where any component
changed; clip_frac = clipn / (steps * n_agents). It is the fraction of (step, agent) pairs with an out-of-range
parameter estimate. It is NOT a thrust-saturation fraction.""")
S20, S40 = SEEDS[:20], SEEDS
def clipv(p, seeds, tr):
    return np.array([ctx.R[("CON-ATT", "s=1", 1.0, float(p), tr)][sd]["clip_frac"] for sd in seeds])
def rm(p, seeds, tr):
    return ctx.vec(("CON-ATT", p), 1, 1.0, tr, seeds)
cells2 = {}
with open("results/prereg/cells.csv") as f:
    for r in csv.DictReader(f):
        cells2[(r["method"], r["profile"], float(r["speed"]), float(r["p_drop"]), r["traj"])] = r
print(f"{'comparison':<26}{'traj':<8}{'n':>3}{'lossless':>10}{'dropped':>10}{'diff':>9}{'seeds with clip>0 (L/D)':>26}{'cells.csv(drop)':>17}{'|d|':>9}")
for p, seeds, lab in ((0.9, S20, "p=0.9 vs lossless (20)"), (0.5, S40, "p=0.5 vs lossless (40)")):
    for tr in pa.TRAJS:
        b, c = clipv(0.0, seeds, tr), clipv(p, seeds, tr)
        cc = float(cells2[("CON-ATT", "s=1", 1.0, p, tr)]["clip_frac_mean"])
        dd = abs(c.mean() - cc)
        ok = dd < 5e-7 and int(cells2[("CON-ATT", "s=1", 1.0, p, tr)]["n"]) == len(seeds)
        if not ok:
            FAILED.append(f"(f) clip_frac mean differs from cells.csv: p={p} {tr}")
        print(f"{lab:<26}{tr:<8}{len(seeds):>3}{b.mean():>10.5f}{c.mean():>10.5f}{c.mean() - b.mean():>9.5f}"
              f"{str(int((b > 0).sum())) + '/' + str(int((c > 0).sum())):>26}{cc:>17.5f}{dd:>9.1e}")
print("\nCross-check of the seed set against Table IV (tracking error, mean RMSE_eff cm, 20 seeds, lossless vs p=0.9):")
for tr in pa.TRAJS:
    print(f"    {tr:<8} lossless {rm(0.0, S20, tr).mean():.3f}   p=0.9 {rm(0.9, S20, tr).mean():.3f}")
print("(paper Table IV: tracking error 2.88/2.91 lossless and 2.80/2.90 at p=0.9; clip_frac is a mean over episodes.)")

banner("Summary")
print("Cross-check failures:", FAILED if FAILED else "none")
print(f"Total run time: {time.time() - T0:.1f} s")
sys.exit(1 if FAILED else 0)
