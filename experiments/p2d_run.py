"""P2d（規格 docs/notes/p2d_prespec.md）：調參 / 雜訊底線 / 感測掃描 / 掉包掃描。全部 pilot。
  p2d_run.py grid                 寫 results/p2d/grid.json（跑之前先寫；已存在就不覆蓋）
  p2d_run.py tune [workers]       階段 1：各方法網格（moderate x1.0，circle+fig8，seeds 2020-2024）
  p2d_run.py ext [workers]        階段 2：選到邊界 -> 延伸一次（寫進 grid.json 的 ext，再跑），然後停；寫 best.json
  p2d_run.py floor [workers]      雜訊底線 seeds 2020-2029（最佳 hp、moderate x1.0）+ CON-ATT 調參 seeds 補跑
  p2d_run.py sweep [workers]      感測掃描 s x 5 方法 x 2 軌跡 x seeds 2200-2219
  p2d_run.py drop [workers]       掉包 p=0.5,0.9（p=0 沿用 sweep s=1）
  p2d_run.py report               -> results/p2d/P2d_report.md + delta_s.png
結果 results/p2d/{tune,floor,sweep,drop}.jsonl（可續跑）。
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, "experiments")
from p2c_common import boot_ci, load, run_jobs

OUT = "results/p2d"
TUNE_SEEDS = list(range(2020, 2025))
FLOOR_SEEDS = list(range(2020, 2030))
SWEEP_SEEDS = list(range(2200, 2220))
TRAJS = ["circle", "fig8"]
S_LIST = [0, 0.5, 1, 1.75, 2.5]
METHODS = ["DOB", "LOC-M", "EKF-L", "CEN-ATT", "CON-ATT"]
LOCAL = ["DOB", "LOC-M", "EKF-L"]
SHARED = ["CEN-ATT", "CON-ATT"]
EST = {"LOC-M", "EKF-L", "CEN-ATT", "CON-ATT", "EKF-L+GEO"}     # 有模型前饋的方法一律用 est 補償（prespec）；DOB 不補償
GRID = {
    "DOB": {"fc": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 1000.0]},
    "LOC-M": {"lam": [0.999, 0.9995, 0.9998, 1.0], "prior": [0.5, 1.0]},
    "CEN-ATT": {"lam": [0.999, 0.9995, 0.9998, 1.0], "prior": [0.5, 1.0]},
    "EKF-L": {"q_b": [5e-4, 2e-3, 8e-3, 3e-2], "r_scale": [1.0, 3.0]},
}
TUNED = list(GRID)
# 延伸規則：各軸的「下一個」值（單軸延伸，其他軸取目前最佳）。lam 上緣 1.0 無法延伸（記為邊界）。fc=1000（不濾波）不算邊界。
EXT_RULE = {"fc": (0.5, 128.0), "lam": (0.998, None), "prior": (0.25, 2.0), "q_b": (1.25e-4, 0.12), "r_scale": (0.5, 10.0)}
NOTE = ("DOB 只調 fc；LOC-M/CEN-ATT 調 lam、prior（prior=先驗標準差整體倍率，基準 prior_dm 0.5、zeta_std 0.3、PRIOR_STD）；"
        "EKF-L 調 q_b（IMU 偏差隨機漫步，模型失配最大）與 r_scale（R 倍率，sigma_f 只含白雜訊）兩個，其餘固定預設。"
        "每方法 8 組。CON-ATT 用 CEN-ATT 的 hp。選擇指標：2020-2024 x circle/fig8 的平均 RMSE（>1 m 或發散記 1 m），平手取成功率高者。")


def grid_points(g):
    ks = list(g)
    return [dict(zip(ks, v)) for v in itertools.product(*[g[k] for k in ks])]


def hk(hp):
    return ",".join(f"{k}={v:g}" for k, v in sorted(hp.items()))


def eta_of(m):
    return "est" if m in EST else "none"


def mk(tag, m, prof, tr, s, hp):
    return dict(key=f"{tag}|{m}|{prof}|{tr}|{s}|{hk(hp)}", tag=tag, method=m, profile=prof, speed=1.0, traj=tr, seed=s, hp=hp,
                eta_comp=eta_of(m))


def gridfile():
    return json.load(open(f"{OUT}/grid.json"))


def tune_jobs(points):
    return [mk("tune", m, "moderate", tr, s, hp) for m, hps in points.items() for hp in hps for tr in TRAJS for s in TUNE_SEEDS]


def rm(r):
    v = r["rmse"]
    return 1.0 if (v is None or not np.isfinite(v)) else min(float(v), 1.0)


def tune_table():
    """(method, hk) -> (mean rmse, success rate, hp)，只看 tag=tune、moderate x1.0、2020-2024。"""
    acc = {}
    for r in load(f"{OUT}/tune.jsonl"):
        if r["tag"] != "tune":
            continue
        a = acc.setdefault((r["method"], hk(r["hp"])), dict(hp=r["hp"], x=[], ok=[]))
        a["x"].append(rm(r)); a["ok"].append(bool(r["success"]))
    return {k: (float(np.mean(v["x"])), float(np.mean(v["ok"])), v["hp"], len(v["x"])) for k, v in acc.items()}


def best_of(method, T, allowed=None):
    c = [(v[0], -v[1], k, v[2]) for k, v in T.items() if k[0] == method and v[3] == 10 and (allowed is None or k[1] in allowed)]
    return min(c)


def boundary(method, hp, g):
    """回傳 [(軸, 'lo'|'hi')]：最佳點落在該軸已測範圍的端點（fc=1000 不算上緣）。"""
    out = []
    for ax, vals in g.items():
        vs = [v for v in vals if not (ax == "fc" and v >= 1000)]
        if hp[ax] == min(vs):
            out.append((ax, "lo"))
        if hp[ax] == max(vs) and len(vs) > 1:
            out.append((ax, "hi"))
    return out


def cmd_grid():
    p = f"{OUT}/grid.json"
    if os.path.exists(p):
        print("grid.json 已存在，不覆蓋"); return
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(note=NOTE, seeds=TUNE_SEEDS, trajs=TRAJS, condition="moderate x1.0", grid=GRID,
                   points={m: grid_points(g) for m, g in GRID.items()}, ext_rule=EXT_RULE, ext={}), open(p, "w"), indent=1)
    print({m: len(grid_points(g)) for m, g in GRID.items()})


def cmd_tune(w):
    G = gridfile()
    run_jobs(tune_jobs(G["points"]), f"{OUT}/tune.jsonl", w)


def cmd_ext(w):
    G = gridfile()
    T = tune_table()
    ext = {}
    for m in TUNED:
        _, _, _, hp = best_of(m, T)
        new = []
        for ax, side in boundary(m, hp, G["grid"][m]):
            v = EXT_RULE[ax][0 if side == "lo" else 1]
            if v is None:
                continue
            new.append({**hp, ax: v})
        ext[m] = dict(stage1_best=hp, boundary=[f"{a}:{s}" for a, s in boundary(m, hp, G["grid"][m])], points=new)
    G["ext"] = ext
    json.dump(G, open(f"{OUT}/grid.json", "w"), indent=1)
    print(json.dumps(ext, indent=1))
    run_jobs(tune_jobs({m: e["points"] for m, e in ext.items()}), f"{OUT}/tune.jsonl", w)
    T = tune_table()
    best = {}
    for m in TUNED:
        sc, nf, k, hp = best_of(m, T)
        best[m] = dict(hp=hp, tune_rmse_cm=sc * 100, tune_succ=-nf,
                       still_boundary=[f"{a}:{s}" for a, s in boundary(m, hp, {ax: sorted({q[ax] for q in G["points"][m] + ext[m]["points"]}) for ax in G["grid"][m]})])
    best["CON-ATT"] = dict(hp=dict(best["CEN-ATT"]["hp"]), note="與 CEN-ATT 共用 hp")
    json.dump(best, open(f"{OUT}/best.json", "w"), indent=1)
    print(json.dumps(best, indent=1))


def cmd_ext2(w, pts):
    """使用者核准的偏離（2026-10-30）：EKF-L 的 r_scale 繼續往外延伸（prespec 原為只延伸一次）。pts = JSON hp 列表。"""
    G = gridfile()
    G.setdefault("ext2", {"note": "偏離 p2d_prespec：使用者核准 EKF-L r_scale 延伸到不在邊界為止；條件同原調參", "points": []})
    for q in pts:
        if q not in G["ext2"]["points"]:
            G["ext2"]["points"].append(q)
    json.dump(G, open(f"{OUT}/grid.json", "w"), indent=1)
    run_jobs(tune_jobs({"EKF-L": pts}), f"{OUT}/tune.jsonl", w)
    T = tune_table()
    for k, v in sorted(T.items(), key=lambda kv: kv[1][0]):
        if k[0] == "EKF-L":
            print(f"{k[1]:28s} {v[0]*100:.3f} cm succ {v[1]:.2f} n={v[3]}")


def cmd_rebest():
    """依 tune.jsonl 全部點重選 EKF-L 最佳（其他方法不動）；舊 best.json 備份成 best_v1.json（只備份一次）。"""
    import shutil
    if not os.path.exists(f"{OUT}/best_v1.json"):
        shutil.copy(f"{OUT}/best.json", f"{OUT}/best_v1.json")
    G, T = gridfile(), tune_table()
    best = json.load(open(f"{OUT}/best_v1.json"))
    sc, nf, k, hp = best_of("EKF-L", T)
    seen = [v[2] for kk, v in T.items() if kk[0] == "EKF-L"]
    axes = {ax: sorted({q[ax] for q in seen}) for ax in G["grid"]["EKF-L"]}
    best["EKF-L"] = dict(hp=hp, tune_rmse_cm=sc * 100, tune_succ=-nf, still_boundary=[f"{a}:{s}" for a, s in boundary("EKF-L", hp, axes)])
    json.dump(best, open(f"{OUT}/best.json", "w"), indent=1)
    print(json.dumps(best["EKF-L"], indent=1))


def best_hp():
    return {m: v["hp"] for m, v in json.load(open(f"{OUT}/best.json")).items()}


def cmd_floor(w):
    B = best_hp()
    J = [mk("floor", m, "moderate", tr, s, B[m]) for m in METHODS for tr in TRAJS for s in FLOOR_SEEDS]
    run_jobs(J, f"{OUT}/floor.jsonl", w)


def cmd_sweep(w):
    B = best_hp()
    J = [mk("sweep", m, f"s={s:g}", tr, sd, B[m]) for s in S_LIST for m in METHODS for tr in TRAJS for sd in SWEEP_SEEDS]
    run_jobs(J, f"{OUT}/sweep.jsonl", w)


def cmd_drop(w):
    B = best_hp()
    J = [mk("drop", "CON-ATT", "s=1", tr, sd, {**B["CON-ATT"], "p_drop": p}) for p in (0.5, 0.9) for tr in TRAJS for sd in SWEEP_SEEDS]
    run_jobs(J, f"{OUT}/drop.jsonl", w)


# ------------------------------------------------------------------ report
def Z(p):   # 標準常態分位數（Acklam 近似的牛頓修正；避免依賴 scipy）
    from math import erf, exp, pi, sqrt
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def n_needed(sd, delta, alpha=0.05, power=0.8):
    """配對 t 近似：n = ((z_{1-a/2}+z_power) sd/Δ)^2 + z_{1-a/2}^2/2（Guenther 修正）。"""
    za, zb = Z(1 - alpha / 2), Z(power)
    return int(np.ceil(((za + zb) * sd / delta) ** 2 + za ** 2 / 2))


def table(path, tag=None):
    """(method, profile, traj, extra) -> {seed: (rmse, success)}；只取 hp 等於 best.json 的列（舊 hp 的列保留在檔內不用）。"""
    D = {}
    B = best_hp()
    for r in load(path):
        hh = {k: v for k, v in r["hp"].items() if k != "p_drop"}
        if hh != B[r["method"]]:
            continue
        D.setdefault((r["method"], r["profile"], r["traj"], hk({k: v for k, v in r["hp"].items() if k == "p_drop"})), {})[r["seed"]] = (rm(r), bool(r["success"]))
    return D


def paired(D, a, b, prof, seeds, trajs=TRAJS, ea="", eb=""):
    """每 seed 的差 (a-b)，單位 cm；trajs 多於 1 條時先對軌跡平均。"""
    out = []
    for s in seeds:
        out.append(np.mean([D[(a, prof, t, ea)][s][0] - D[(b, prof, t, eb)][s][0] for t in trajs]) * 100)
    return np.array(out)


def ci(x):
    m, lo, hi = boot_ci(x)
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def cmd_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = ["# P2d 報告（pilot；規格 docs/notes/p2d_prespec.md；全部是 pilot，不當正式結論）\n"]
    G, B = gridfile(), json.load(open(f"{OUT}/best.json"))
    T = tune_table()
    # ---- 1 調參
    L.append("## 1. 調參（moderate x1.0；seeds 2020-2024 x circle/fig8；平均 RMSE，>1 m/發散記 1 m）\n")
    L.append("選到邊界時延伸一次後停。表列各方法所有試過的點（RMSE cm / 成功率）。\n")
    for m in TUNED:
        rows = sorted([(v[0], v[1], v[2]) for k, v in T.items() if k[0] == m])
        L.append(f"**{m}**：選定 {hk(B[m]['hp'])}（{B[m]['tune_rmse_cm']:.2f} cm，成功率 {B[m]['tune_succ']:.2f}）；"
                 f"階段 1 邊界：{G['ext'][m]['boundary'] or '無'}；延伸後仍在邊界：{B[m]['still_boundary'] or '無'}\n")
        L.append("| hp | RMSE cm | 成功率 |\n|---|---|---|")
        for sc, su, hp in rows:
            L.append(f"| {hk(hp)}{' **' if hp == B[m]['hp'] else ''} | {sc*100:.2f} | {su:.2f} |")
        L.append("")
        if m == "EKF-L" and G.get("ext2"):
            L.append("**偏離（使用者核准，2026-10-30）**：p2d_prespec 原規定選到邊界只延伸一次；EKF-L 的 r_scale 因此再延伸（新增 30、100，及在 r_scale=10 對 q_b 補 2e-3、1.25e-4 共 4 點）。"
                     "r_scale=10 兩側都有點（3 和 30），q_b 兩側也有點，最佳不在邊界；最佳 hp 沒變，故 EKF-L 的底線、掃描、掉包結果不需重跑。舊版報告見 P2d_report_v1.md。\n")
    L.append(f"CON-ATT：共用 CEN-ATT 的 hp（{hk(B['CON-ATT']['hp'])}）。\n")
    # ---- 2 floor
    F = table(f"{OUT}/floor.jsonl")
    Tt = {}
    for m in METHODS:   # 調參 seeds（2020-2024）選最佳；CON-ATT 的調參分數來自 floor.jsonl 的 2020-2024
        Tt[m] = np.mean([F[(m, "moderate", t, "")][s][0] for t in TRAJS for s in TUNE_SEEDS])
    bs = min(SHARED, key=lambda m: Tt[m]); bl = min(LOCAL, key=lambda m: Tt[m])
    L.append("## 2. 雜訊底線（moderate x1.0；seeds 2020-2029，n=10；注意 2020-2024 同時是調參 seeds）\n")
    L.append(f"調參 seeds（2020-2024）RMSE cm：" + "；".join(f"{m} {Tt[m]*100:.2f}" for m in METHODS) + f"。最佳分享 = **{bs}**，最佳本地 = **{bl}**。\n")
    L.append("| 方法 | circle RMSE cm | fig8 RMSE cm | 兩軌平均 | 成功率 |\n|---|---|---|---|---|")
    for m in METHODS:
        v = [np.mean([F[(m, "moderate", t, "")][s][0] for s in FLOOR_SEEDS]) * 100 for t in TRAJS]
        su = np.mean([F[(m, "moderate", t, "")][s][1] for t in TRAJS for s in FLOOR_SEEDS])
        L.append(f"| {m} | {v[0]:.2f} | {v[1]:.2f} | {np.mean(v):.2f} | {su:.2f} |")
    L.append("\n主比較配對差 = 分享 − 本地（cm，負 = 分享較好）的 seed 間標準差：\n")
    L.append("| 比較 | 範圍 | 平均差 [CI] | sd | Δ=0.5 | Δ=0.75 | Δ=1.0 | Δ=本地RMSE 10% (cm) |\n|---|---|---|---|---|---|---|---|")
    pairs = [(bs, bl)] + [(a, b) for a in SHARED for b in LOCAL if (a, b) != (bs, bl)]
    for a, b in pairs:
        for lab, trs in (("pooled(兩軌平均/seed)", TRAJS), ("circle", ["circle"]), ("fig8", ["fig8"])):
            d = paired(F, a, b, "moderate", FLOOR_SEEDS, trs)
            sd = d.std(ddof=1)
            lr = np.mean([F[(b, "moderate", t, "")][s][0] for t in trs for s in FLOOR_SEEDS]) * 100
            d10 = max(0.5, 0.1 * lr) if False else 0.1 * lr
            ns = [n_needed(sd, x) for x in (0.5, 0.75, 1.0)]
            n10 = n_needed(sd, d10)
            mark = "**" if (a, b) == (bs, bl) else ""
            L.append(f"| {mark}{a} − {b}{mark} | {lab} | {ci(d)} | {sd:.2f} | {ns[0]} | {ns[1]} | {ns[2]} | {n10}（Δ={d10:.2f}） |")
    L.append("\n註：n = ((z_{0.975}+z_{0.8}) sd/Δ)^2 + z_{0.975}^2/2（配對 t 的常態近似加 Guenther 修正；n 是每格的 seed 數，每格＝單一軌跡）。"
             "「pooled」列的 sd 是兩軌跡先平均後的 seed 間 sd，只在把兩軌跡合併成一格時適用。暫定 Δ = max(0.5 cm, 本地 RMSE 10%)，見 Δ=10% 欄。\n")
    # ---- 3 sweep
    S = table(f"{OUT}/sweep.jsonl")
    P = lambda s: f"s={s:g}"
    L.append(f"## 3. 感測強度掃描（seeds 2200-2219，n=20；最佳分享={bs}、最佳本地={bl}，由調參 seeds 選定）\n")
    L.append("RMSE cm（兩軌跡平均）/ 成功率（兩軌跡 40 集）：\n")
    L.append("| s | " + " | ".join(METHODS) + " |\n|---|" + "---|" * len(METHODS))
    for s in S_LIST:
        row = []
        for m in METHODS:
            v = np.mean([S[(m, P(s), t, "")][sd][0] for t in TRAJS for sd in SWEEP_SEEDS]) * 100
            su = np.mean([S[(m, P(s), t, "")][sd][1] for t in TRAJS for sd in SWEEP_SEEDS])
            row.append(f"{v:.2f} / {su:.2f}")
        L.append(f"| {s:g} | " + " | ".join(row) + " |")
    L.append("\n分軌跡 RMSE cm：\n")
    L.append("| s | 軌跡 | " + " | ".join(METHODS) + " |\n|---|---|" + "---|" * len(METHODS))
    for s in S_LIST:
        for t in TRAJS:
            L.append(f"| {s:g} | {t} | " + " | ".join(f"{np.mean([S[(m, P(s), t, '')][sd][0] for sd in SWEEP_SEEDS])*100:.2f}" for m in METHODS) + " |")
    L.append(f"\n**Δ(s) = {bs} − {bl}**（cm；負 = 分享較好；seed 配對 bootstrap 95% CI）：\n")
    L.append("| s | pooled | circle | fig8 |\n|---|---|---|---|")
    curves = {}
    for s in S_LIST:
        cells = []
        for lab, trs in (("pooled", TRAJS), ("circle", ["circle"]), ("fig8", ["fig8"])):
            d = paired(S, bs, bl, P(s), SWEEP_SEEDS, trs)
            curves.setdefault(lab, []).append(boot_ci(d))
            cells.append(ci(d))
        L.append(f"| {s:g} | " + " | ".join(cells) + " |")
    L.append("\n**CON-ATT − CEN-ATT**（cm）：\n")
    L.append("| s | pooled | circle | fig8 |\n|---|---|---|---|")
    for s in S_LIST:
        L.append(f"| {s:g} | " + " | ".join(ci(paired(S, "CON-ATT", "CEN-ATT", P(s), SWEEP_SEEDS, trs)) for trs in (TRAJS, ["circle"], ["fig8"])) + " |")
    L.append("\n補充（事後；不屬於主要量）：各分享法對各本地法的 pooled 差：\n")
    L.append("| s | " + " | ".join(f"{a} − {b}" for a in SHARED for b in LOCAL) + " |\n|---|" + "---|" * 6)
    for s in S_LIST:
        L.append(f"| {s:g} | " + " | ".join(ci(paired(S, a, b, P(s), SWEEP_SEEDS)) for a in SHARED for b in LOCAL) + " |")
    # 分界
    bd = []
    for lab in ("pooled", "circle", "fig8"):
        xs = [(s, c) for s, c in zip(S_LIST, curves[lab])]
        first = next((s for s, c in xs if c[2] < 0), None)
        bd.append(f"{lab}：{'CI 上界 <0 最小的 s = %g' % first if first is not None else '沒有任何 s 的 CI 完全 <0'}"
                  f"；最佳本地反超（CI 下界 >0）的 s = {[s for s, c in xs if c[1] > 0] or '無'}")
    L.append("\n**分界（Δ(s) 的 CI 完全 <0 的最小 s）**：" + "；".join(bd) + "\n")
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for lab, col in (("circle", "#1f77b4"), ("fig8", "#d95f02")):
        c = np.array(curves[lab])
        ax.plot(S_LIST, c[:, 0], "o-", color=col, label=lab)
        ax.fill_between(S_LIST, c[:, 1], c[:, 2], color=col, alpha=0.2)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("sensing scale s (0=clean, 1=moderate, 2.5≈harsh)")
    ax.set_ylabel(f"Δ(s) = {bs} − {bl}  RMSE (cm)")
    ax.set_title("Shared minus best local (negative = sharing better)")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/delta_s.png", dpi=130)
    L.append(f"![Δ(s)](delta_s.png)\n")
    # ---- 4 drop
    Dr = table(f"{OUT}/drop.jsonl")
    L.append("## 4. 掉包掃描（CON-ATT；moderate x1.0；seeds 2200-2219；p=0 沿用 sweep s=1）\n")
    allD = {**{k: v for k, v in S.items() if k[0] == "CON-ATT" and k[1] == "s=1"}}
    def get(p, t):
        return S[("CON-ATT", "s=1", t, "")] if p == 0 else Dr[("CON-ATT", "s=1", t, f"p_drop={p:g}")]
    L.append("| p_drop | CON RMSE cm（circle / fig8 / 平均） | 成功率 | CON − LOC-M(est) pooled [CI] | CON − EKF-L(est) pooled [CI] | circle 對 LOC-M / EKF-L | fig8 對 LOC-M / EKF-L |\n|---|---|---|---|---|---|---|")
    for p in (0, 0.5, 0.9):
        v = [np.mean([get(p, t)[sd][0] for sd in SWEEP_SEEDS]) * 100 for t in TRAJS]
        su = np.mean([get(p, t)[sd][1] for t in TRAJS for sd in SWEEP_SEEDS])
        def dd(b, trs):
            return np.array([np.mean([get(p, t)[sd][0] - S[(b, "s=1", t, "")][sd][0] for t in trs]) * 100 for sd in SWEEP_SEEDS])
        L.append(f"| {p:g} | {v[0]:.2f} / {v[1]:.2f} / {np.mean(v):.2f} | {su:.2f} | {ci(dd('LOC-M', TRAJS))} | {ci(dd('EKF-L', TRAJS))} | "
                 f"{ci(dd('LOC-M', ['circle']))} ; {ci(dd('EKF-L', ['circle']))} | {ci(dd('LOC-M', ['fig8']))} ; {ci(dd('EKF-L', ['fig8']))} |")
    L.append("\n## 5. 偏離與注意事項\n見作者的內部紀錄（未公開）。\n")
    open(f"{OUT}/P2d_report.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


# ------------------------------------------------------------------ EKF-L+GEO（拆解對照；PREREG 前加入）
GEO = "EKF-L+GEO"
GEO_GRID = {"q_b": [5e-4, 2e-3], "r_scale": [3.0, 10.0], "use_dc": [0, 1]}    # 8 組；use_dc 是結構選項（不算邊界軸）
GEO_EXT = {"q_b": 4.0, "r_scale": 3.0}                                           # 邊界軸往外一步的倍率
GEO_S = [0, 1, 2.5]


def geo_table():
    return {k: v for k, v in tune_table().items() if k[0] == GEO}


def cmd_geo_tune(w):
    """階段 1：8 組；選到邊界（q_b、r_scale 兩軸）就沿該軸往外一步（其他軸取目前最佳）再跑，直到不在邊界。結果追加到 tune.jsonl，更新 best.json。"""
    import shutil
    G = gridfile()
    G.setdefault("geo", dict(note="EKF-L+GEO 拆解對照調參（條件同其他方法：moderate x1.0、seeds 2020-2024、circle+fig8）；use_dc 是結構選項", grid=GEO_GRID, rounds=[]))
    pts = grid_points(GEO_GRID)
    G["geo"]["rounds"].append(dict(stage="stage1", points=pts))
    json.dump(G, open(f"{OUT}/grid.json", "w"), indent=1)
    run_jobs(tune_jobs({GEO: pts}), f"{OUT}/tune.jsonl", w)
    for rnd in range(8):
        T = geo_table()
        sc, nf, k, hp = best_of(GEO, T)
        seen = [v[2] for v in T.values()]
        axes = {ax: sorted({q[ax] for q in seen}) for ax in GEO_EXT}
        bd = boundary(GEO, hp, axes)
        print(f"round {rnd}: best {hk(hp)} {sc*100:.3f} cm; boundary {bd}", flush=True)
        if not bd:
            break
        new = []
        for ax, side in bd:      # 往外一步：q_b 乘／除 4，r_scale 乘／除 3（第一輪即 1.25e-4 / 8e-3 / 1.0 / 30）
            f_ = GEO_EXT[ax]
            q = {**hp, ax: (min(axes[ax]) / f_ if side == "lo" else max(axes[ax]) * f_)}
            if hk(q) not in {hk(x) for x in seen}:
                new.append(q)
        if not new:
            print("邊界軸已無可延伸的新點"); break
        G = gridfile(); G["geo"]["rounds"].append(dict(stage=f"ext{rnd+1}", points=new)); json.dump(G, open(f"{OUT}/grid.json", "w"), indent=1)
        run_jobs(tune_jobs({GEO: new}), f"{OUT}/tune.jsonl", w)
    T = geo_table()
    sc, nf, k, hp = best_of(GEO, T)
    seen = [v[2] for v in T.values()]
    axes = {ax: sorted({q[ax] for q in seen}) for ax in GEO_EXT}
    if not os.path.exists(f"{OUT}/best_v2.json"):
        shutil.copy(f"{OUT}/best.json", f"{OUT}/best_v2.json")
    best = json.load(open(f"{OUT}/best.json"))
    best[GEO] = dict(hp=hp, tune_rmse_cm=sc * 100, tune_succ=-nf, still_boundary=[f"{a}:{s_}" for a, s_ in boundary(GEO, hp, axes)])
    json.dump(best, open(f"{OUT}/best.json", "w"), indent=1)
    for k_, v in sorted(T.items(), key=lambda kv: kv[1][0]):
        print(f"{k_[1]:36s} {v[0]*100:.3f} cm succ {v[1]:.2f} n={v[3]}")
    print(json.dumps(best[GEO], indent=1))


def cmd_geo_pilot(w):
    B = best_hp()
    J = [mk("sweep", GEO, f"s={s:g}", tr, sd, B[GEO]) for s in GEO_S for tr in TRAJS for sd in SWEEP_SEEDS]
    run_jobs(J, f"{OUT}/sweep.jsonl", w)


def cmd_geo_report():
    S = table(f"{OUT}/sweep.jsonl")
    P = lambda s: f"s={s:g}"
    L = ["# EKF-L+GEO 拆解 pilot（s ∈ {0,1,2.5} x seeds 2200-2219 x circle/fig8；RMSE cm，差 = 前 − 後，負 = 前者較好；95% bootstrap CI）\n",
         f"EKF-L+GEO hp：{hk(best_hp()[GEO])}\n"]
    L.append("| s | EKF-L | EKF-L+GEO | CON-ATT | 成功率 GEO |\n|---|---|---|---|---|")
    for s in GEO_S:
        L.append(f"| {s:g} | " + " | ".join(f"{np.mean([S[(m, P(s), t, '')][sd][0] for t in TRAJS for sd in SWEEP_SEEDS])*100:.2f}" for m in ("EKF-L", GEO, "CON-ATT"))
                 + f" | {np.mean([S[(GEO, P(s), t, '')][sd][1] for t in TRAJS for sd in SWEEP_SEEDS]):.2f} |")
    for title, a, b in (("隊友位置（幾何）的收益：EKF-L+GEO − EKF-L", GEO, "EKF-L"), ("估計共享的收益：CON-ATT − EKF-L+GEO", "CON-ATT", GEO),
                        ("總收益：CON-ATT − EKF-L", "CON-ATT", "EKF-L")):
        L.append(f"\n**{title}**（cm）\n\n| s | pooled | circle | fig8 |\n|---|---|---|---|")
        for s in GEO_S:
            L.append(f"| {s:g} | " + " | ".join(ci(paired(S, a, b, P(s), SWEEP_SEEDS, trs)) for trs in (TRAJS, ["circle"], ["fig8"])) + " |")
    open(f"{OUT}/geo_report.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    c = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    {"grid": cmd_grid, "tune": lambda: cmd_tune(w), "ext": lambda: cmd_ext(w), "floor": lambda: cmd_floor(w), "sweep": lambda: cmd_sweep(w),
     "drop": lambda: cmd_drop(w), "report": cmd_report, "ext2": lambda: cmd_ext2(w, json.loads(sys.argv[3])),
     "rebest": cmd_rebest, "geo_tune": lambda: cmd_geo_tune(w), "geo_pilot": lambda: cmd_geo_pilot(w), "geo_report": cmd_geo_report}[c]()
