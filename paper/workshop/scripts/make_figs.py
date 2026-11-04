"""Figures for the workshop paper. Run from repo root: .venv/bin/python paper/workshop/scripts/make_figs.py
Fig 1 from results/prereg/tests.csv (formal values); Fig 2 from results/prereg/episodes.jsonl (raw RMSE, own bootstrap)."""
import csv, json, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = [0, 0.5, 1, 1.75, 2.5]
rows = [r for r in csv.DictReader(open("results/prereg/tests.csv")) if r["family"] == "F1"]
def get(h, t):
    d = {float(r["s"]): r for r in rows if r["hyp"] == h and r["traj"] == t}
    return [(float(d[s]["mean_diff_cm"]), float(d[s]["ci95_lo"]), float(d[s]["ci95_hi"])) for s in S]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
for a, t in zip(ax, ["circle", "fig8"]):
    for h, c, lab, off in [("H1a", "C0", r"$D_a$: EKF-L+GEO $-$ EKF-L", -0.04), ("H1b", "C1", r"$D_b$: CON-ATT $-$ EKF-L+GEO", 0.04)]:
        v = np.array(get(h, t)); x = np.array(S) + off
        m = v[:, 0]; lo = np.maximum(v[:, 1], -1.4); hi = v[:, 2]
        a.errorbar(x, np.maximum(m, -1.4), yerr=[np.maximum(m, -1.4) - lo, hi - np.maximum(m, -1.4)], fmt="o-", color=c, ms=3.5, capsize=2, lw=1, label=lab)
        for xi, mi in zip(x, m):
            if mi < -1.4: a.annotate(f"{mi:.2f}\n(off scale)", (xi, -1.4), xytext=(xi - 0.9, -1.15), fontsize=6, arrowprops=dict(arrowstyle="->", lw=0.5))
    a.axhline(0, color="gray", lw=0.5); a.axhline(-0.5, color="gray", lw=0.5, ls=":")
    a.set_title(t, fontsize=8); a.set_xlabel("sensing strength $s$", fontsize=8); a.set_ylim(-1.45, 0.6); a.tick_params(labelsize=7)
ax[0].set_ylabel("paired difference (cm, 95% CI)", fontsize=8); ax[0].legend(fontsize=6, loc="lower left")
plt.tight_layout(); plt.savefig("paper/workshop/figs/fig_H1.pdf"); plt.close()

E = collections.defaultdict(dict)
for l in open("results/prereg/episodes.jsonl"):
    r = json.loads(l)
    if r["speed"] != 1.0 or r["method"] not in ("DOB", "CON-ATT"): continue
    if r["method"] == "CON-ATT" and r["hp"].get("p_drop", 0) != 0: continue
    E[(r["method"], r["profile"], r["traj"])][r["seed"]] = 100 * r["rmse"]
rng = np.random.default_rng(0)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)
for a, t in zip(ax, ["circle", "fig8"]):
    ms, los, his = [], [], []
    for s in S:
        d, c = E[("DOB", f"s={s:g}", t)], E[("CON-ATT", f"s={s:g}", t)]
        x = np.array([d[k] - c[k] for k in sorted(d)])
        idx = rng.integers(0, len(x), (10000, len(x))); bm = x[idx].mean(1)
        ms.append(x.mean()); los.append(np.percentile(bm, 2.5)); his.append(np.percentile(bm, 97.5))
    ms, los, his = map(np.array, (ms, los, his))
    a.errorbar(S, ms, yerr=[ms - los, his - ms], fmt="o-", ms=3.5, capsize=2, lw=1)
    a.axhline(0, color="gray", lw=0.5); a.set_title(t, fontsize=8); a.set_xlabel("sensing strength $s$", fontsize=8); a.tick_params(labelsize=7)
ax[0].set_ylabel("DOB $-$ CON-ATT, raw RMSE (cm)", fontsize=8)
plt.tight_layout(); plt.savefig("paper/workshop/figs/fig_H2_raw.pdf"); plt.close()
