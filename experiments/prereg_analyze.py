"""PREREG 正式分析腳本（只依 docs/PREREG.md §1、§4、§5、§6、§10 寫成）。

  python experiments/prereg_analyze.py                       # 正式：results/prereg/episodes.jsonl，seeds 3000:3040，缺資料即報錯停止
  python experiments/prereg_analyze.py --input results/p2d/sweep.jsonl --seeds 2200:2220 --allow-partial --out .scratch/prereg_dry
      # dry-run：缺格子略過並列出（不是結論）

輸出（--out，預設 results/prereg/）：cells.csv、tests.csv、cells_hyp.csv、h2_h4_slopes.csv、sensitivity.csv、
PREREG_results.md、fig_H1_by_s.png、fig_H2_B_DOB.png。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

S_LIST = [0, 0.5, 1, 1.75, 2.5]
TRAJS = ["circle", "fig8"]
DELTA = 0.5          # cm，實質門檻
TOST_B = 0.25        # cm
TOST_B2 = 0.5        # 敏感度
ALPHA = 0.05
B_PERM = 100_000
B_BOOT = 10_000
TIE_TOL = 1e-9

FROZEN = {
    "DOB": {"fc": 16.0},
    "LOC-M": {"lam": 0.998, "prior": 1.0},
    "EKF-L": {"q_b": 5e-4, "r_scale": 10.0},
    "EKF-L+GEO": {"q_b": 5e-4, "r_scale": 10.0, "use_dc": 0},
    "CEN-ATT": {"lam": 0.9998, "prior": 2.0},
    "CON-ATT": {"lam": 0.9998, "prior": 2.0, "iters": 2},
    "ORACLE": {},
}
FROZEN_HYB = {1.0: {"fc": 8.0, "tau": 0.07}, 1.5: {"fc": 1000.0, "tau": 0.07}}
HP_DEFAULT = {"iters": 2}   # 缺欄位時視為此預設（pilot 的 CON-ATT 沒記 iters/p_drop）


class Missing(Exception):
    pass


# ---------------------------------------------------------------- 基本統計函式
def rmse_eff(r, p2d_rule=False):
    """§4：失敗/發散/非有限/ >1 m 記 100 cm。p2d_rule：只對不有限或 >1 m 記 100 cm（pilot rm()）。"""
    v = r.get("rmse")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 100.0
    if (not r.get("finite", True)) or (not math.isfinite(v)):
        return 100.0
    if not p2d_rule and not r.get("success", False):
        return 100.0
    if v > 1.0:
        return 100.0
    return 100.0 * v


_SIGN = {}
_BOOT = {}


def sign_matrix(n):
    if n not in _SIGN:
        _SIGN[n] = (np.random.default_rng(0).integers(0, 2, (B_PERM, n)) * 2 - 1).astype(np.float64)
    return _SIGN[n]


def boot_idx(n):
    if n not in _BOOT:
        _BOOT[n] = np.random.default_rng(0).integers(0, n, (B_BOOT, n))
    return _BOOT[n]


def signflip_p(x, S=None):
    """單尾（H0: 均值 >= 0，備擇: 均值 < 0）：(1 + #{m_b <= mean}) / (B + 1)。"""
    x = np.asarray(x, float)
    S = sign_matrix(len(x)) if S is None else S
    mb = S @ x / len(x)
    m = x.mean()
    tol = TIE_TOL * max(1.0, float(np.abs(x).max()))
    return float((1 + np.sum(mb <= m + tol)) / (S.shape[0] + 1))


def boot_ci(x, level=95):
    x = np.asarray(x, float)
    mb = x[boot_idx(len(x))].mean(1)
    a = (100 - level) / 2
    return float(np.percentile(mb, a)), float(np.percentile(mb, 100 - a))


def holm(ps, alpha=ALPHA):
    """回傳 (調整後 p, 是否拒絕)。調整 p = cummax((m-i)·p_(i))，上限 1。"""
    ps = np.asarray(ps, float)
    m = len(ps)
    order = np.argsort(ps, kind="stable")
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj, adj < alpha


def tost(x, bound=TOST_B):
    """p_TOST = max(p1, p2)；p1: x-bound 的單尾（H0 均值 >= bound）；p2: -(x+bound) 的單尾（H0 均值 <= -bound）。"""
    x = np.asarray(x, float)
    p1 = signflip_p(x - bound)
    p2 = signflip_p(-(x + bound))
    lo, hi = boot_ci(x, 90)
    return max(p1, p2), p1, p2, lo, hi


def ols_slope(Y, s=S_LIST):
    """Y: (n, 5)。斜率 = Σ(s-s̄)y / Σ(s-s̄)²。"""
    s = np.asarray(s, float)
    d = s - s.mean()
    return np.asarray(Y, float) @ d / (d @ d)


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ---------------------------------------------------------------- 資料載入
def hp_matches(method, speed, hp):
    exp = dict(FROZEN_HYB[speed]) if method == "HYB" else FROZEN.get(method)
    if exp is None:
        return False
    hp = dict(hp or {})
    hp.pop("p_drop", None)
    for k, v in HP_DEFAULT.items():
        if method == "CON-ATT":
            hp.setdefault(k, v)
    return set(hp) == set(exp) and all(abs(float(hp[k]) - float(v)) < 1e-12 for k, v in exp.items())


def load_rows(path, seeds):
    seen, rows, dup, bad_hp, out_seed = set(), [], 0, 0, 0
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        if r["key"] in seen:
            dup += 1
            continue                      # 同 key 取第一筆
        seen.add(r["key"])
        if r["seed"] not in seeds:
            out_seed += 1
            continue
        if not hp_matches(r["method"], float(r["speed"]), r.get("hp")):
            bad_hp += 1
            continue
        rows.append(r)
    print(f"讀入 {len(rows)} 集（重複 key {dup}、seed 不在範圍 {out_seed}、hp 不符 {bad_hp} 已略過）")
    R = {}
    for r in rows:
        k = (r["method"], r["profile"], float(r["speed"]), float((r.get("hp") or {}).get("p_drop", 0)), r["traj"])
        R.setdefault(k, {})[r["seed"]] = r
    return R


class Ctx:
    def __init__(self, R, seeds):
        self.R, self.seeds = R, list(seeds)

    def vec(self, A, s, speed, traj, seeds=None, p2d=False, fn=None):
        m, p = A
        k = (m, f"s={s:g}", float(speed), float(p), traj)
        d = self.R.get(k)
        seeds = self.seeds if seeds is None else seeds
        if d is None or any(sd not in d for sd in seeds):
            raise Missing(f"{k} 缺 seeds（有 {0 if d is None else len(d)}/{len(seeds)}）")
        f = fn or (lambda r: rmse_eff(r, p2d))
        return np.array([f(d[sd]) for sd in seeds])

    def diff(self, A, B, s, speed, traj, seeds=None, p2d=False, drop_failed=False):
        a = self.vec(A, s, speed, traj, seeds, p2d)
        b = self.vec(B, s, speed, traj, seeds, p2d)
        if drop_failed:
            sds = self.seeds if seeds is None else seeds
            ok = np.array([self.R[(A[0], f"s={s:g}", float(speed), float(A[1]), traj)][sd].get("success", False) and
                           self.R[(B[0], f"s={s:g}", float(speed), float(B[1]), traj)][sd].get("success", False) for sd in sds])
            return (a - b)[ok]
        return a - b

    def slope_Y(self, yfun):
        return np.stack([yfun(s) for s in S_LIST], axis=1)


# ---------------------------------------------------------------- 檢定規格
EKF, GEO, CON, CEN, DOB, LOC = ("EKF-L", 0), ("EKF-L+GEO", 0), ("CON-ATT", 0), ("CEN-ATT", 0), ("DOB", 0), ("LOC-M", 0)


def build_specs(seeds):
    """每個 spec：family, hyp, s, speed, traj, A, B, kind(diff|slope_pos|slope_neg|tost), get(ctx, p2d, drop_failed)->(x, Y)。"""
    sp = []

    def add(**kw):
        kw.setdefault("seeds", None)
        sp.append(kw)

    for tr in TRAJS:
        for s in S_LIST:
            add(family="F1", hyp="H1a", s=s, speed=1.0, traj=tr, A=GEO, B=EKF, kind="diff")
            add(family="F1", hyp="H1b", s=s, speed=1.0, traj=tr, A=CON, B=GEO, kind="diff")
    for tr in TRAJS:
        add(family="F2", hyp="H2", s="slope", speed=1.0, traj=tr, A=DOB, B=CON, kind="slope_pos")
        add(family="F3", hyp="H3", s=1, speed=1.0, traj=tr, A=CON, B=CEN, kind="tost")
        add(family="F4", hyp="H2-LOCM", s="slope", speed=1.0, traj=tr, A=LOC, B=CON, kind="slope_pos")
        add(family="F5", hyp="H4", s="slope", speed=1.0, traj=tr, A=("Da-Db", 0), B=("", 0), kind="slope_neg")
        for s in (0, 0.5, 1.75, 2.5):
            add(family="F6", hyp="H3", s=s, speed=1.0, traj=tr, A=CON, B=CEN, kind="tost")
        add(family="F7", hyp="H1a@1.5", s=1, speed=1.5, traj=tr, A=GEO, B=EKF, kind="diff")
        add(family="F7", hyp="H1b@1.5", s=1, speed=1.5, traj=tr, A=CON, B=GEO, kind="diff")
        add(family="F8", hyp="DROP0.5-vs-EKF-L", s=1, speed=1.0, traj=tr, A=("CON-ATT", 0.5), B=EKF, kind="diff")
    # 描述性（只報點估計與 CI）
    for tr in TRAJS:
        for s in S_LIST:
            add(family="DESC", hyp="B_DOB", s=s, speed=1.0, traj=tr, A=DOB, B=CON, kind="diff")
            add(family="DESC", hyp="B_LOCM", s=s, speed=1.0, traj=tr, A=LOC, B=CON, kind="diff")
            add(family="DESC", hyp="TOTAL(CON-EKF-L)", s=s, speed=1.0, traj=tr, A=CON, B=EKF, kind="diff")
        add(family="DESC", hyp="DROP0.5-vs-GEO", s=1, speed=1.0, traj=tr, A=("CON-ATT", 0.5), B=GEO, kind="diff")
        add(family="DESC", hyp="DROP0.9-vs-EKF-L", s=1, speed=1.0, traj=tr, A=("CON-ATT", 0.9), B=EKF, kind="diff",
            seeds=list(seeds)[:20])
        add(family="DESC", hyp="DROP0.9-vs-GEO", s=1, speed=1.0, traj=tr, A=("CON-ATT", 0.9), B=GEO, kind="diff",
            seeds=list(seeds)[:20])
    return sp


def get_data(ctx, sp, p2d=False, drop_failed=False):
    """回傳 (x, Y)。diff/tost: x=配對差；slope: x=每 seed 斜率，Y=(n,5)。"""
    k, tr, sd = sp["kind"], sp["traj"], sp["seeds"]
    if k in ("diff", "tost"):
        return ctx.diff(sp["A"], sp["B"], sp["s"], sp["speed"], tr, sd, p2d, drop_failed), None
    h = sp["hyp"]
    if h == "H4":
        Y = ctx.slope_Y(lambda s: ctx.diff(GEO, EKF, s, 1.0, tr, sd, p2d) - ctx.diff(CON, GEO, s, 1.0, tr, sd, p2d))
    else:
        Y = ctx.slope_Y(lambda s: ctx.diff(sp["A"], CON, s, 1.0, tr, sd, p2d))
    return ols_slope(Y), Y


def tid(sp):
    return f"{sp['hyp']}|s={sp['s']}|{sp['traj']}|sp{sp['speed']}"


def run_tests(ctx, specs, allow_partial, p2d=False, drop_failed=False, families=None, tost_bound=TOST_B):
    res, skipped = [], []
    for sp in specs:
        if families and sp["family"] not in families:
            continue
        try:
            x, Y = get_data(ctx, sp, p2d, drop_failed)
        except Missing as e:
            if not allow_partial:
                sys.exit(f"錯誤（缺資料，§10.2 步驟 2 要求停止）：{tid(sp)}：{e}")
            skipped.append((tid(sp), str(e)))
            continue
        r = dict(sp=sp, x=x, Y=Y, n=len(x), mean=float(x.mean()))
        r["ci95"] = boot_ci(x, 95)
        r["ci90"] = boot_ci(x, 90)
        k = sp["kind"]
        if k == "diff" and sp["family"] != "DESC" or k == "slope_neg":
            r["p"] = signflip_p(x)
        elif k == "slope_pos":
            r["p"] = signflip_p(-x)
        elif k == "tost":
            r["p"], r["p1"], r["p2"], _, _ = tost(x, tost_bound)
            r["bound"] = tost_bound
        else:
            r["p"] = float("nan")
        r["rel"] = r["mean"] / ctx_mean(ctx, sp, p2d) if k == "diff" or k == "tost" else float("nan")
        res.append(r)
    for fam in sorted({r["sp"]["family"] for r in res} - {"DESC"}):
        grp = [r for r in res if r["sp"]["family"] == fam]
        adj, rej = holm([r["p"] for r in grp])
        for r, a, j in zip(grp, adj, rej):
            r["p_adj"], r["rej"] = float(a), bool(j)
            if r["sp"]["kind"] == "tost":
                lo, hi = r["ci90"]
                r["rej"] = bool(j and lo > -r["bound"] and hi < r["bound"])
    for r in res:
        if r["sp"]["family"] == "DESC":
            r["p_adj"], r["rej"] = float("nan"), False
    return res, skipped


def ctx_mean(ctx, sp, p2d):
    try:
        b = ctx.vec(sp["B"], sp["s"], sp["speed"], sp["traj"], sp["seeds"], p2d).mean()
    except Missing:
        return float("nan")
    return b if b != 0 else float("nan")


def delta_layer(r):
    sp = r["sp"]
    if not sp["hyp"].startswith(("H1a", "H1b")) or not r["rej"]:
        return ""
    return "達實質門檻" if r["mean"] <= -DELTA else "統計上可偵測但小於Δ"


def reverse(r):
    """§1.2 真值表：該軌跡的 95% CI 完全落在「不利方向」。"""
    lo, hi = r["ci95"]
    return hi < 0 if r["sp"]["kind"] == "slope_pos" else lo > 0


def hyp_evidence(c, f):
    """不成立時的證據層：任一軌跡 CI 反向 -> 反向；恰一軌跡過 Holm -> 單軌成立；其餘 -> 證據不足（TOST 為未等價）。"""
    if c["sp"]["kind"] == "tost":
        return "未等價"
    rv = [r["sp"]["traj"] for r in (c, f) if reverse(r)]
    if rv:
        return "反向(" + ",".join(rv) + ")"
    one = [r["sp"]["traj"] for r in (c, f) if r["rej"]]
    if one:
        return "單軌成立(" + one[0] + ")"
    return "證據不足"


# ---------------------------------------------------------------- 輸出
def fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if (isinstance(v, float) and math.isnan(v)) else f"{float(v):.6g}"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    return v


def write_csv(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt(r.get(c, "")) for c in cols])


def a_name(a):
    return a[0] if a[1] == 0 else f"{a[0]}(p_drop={a[1]:g})"


def cells_table(R):
    rows = []
    for (m, prof, sp_, p, tr), d in sorted(R.items()):
        rs = list(d.values())
        e = np.array([rmse_eff(r) for r in rs])
        lo, hi = boot_ci(e, 95)
        k = sum(bool(r.get("success", False)) for r in rs)
        sl, sh = wilson(k, len(rs))

        def mean_of(key):
            v = [float(r[key]) for r in rs if r.get(key) is not None and np.isfinite(float(r[key]))]
            return float(np.mean(v)) if v else float("nan")
        rows.append(dict(method=m, profile=prof, speed=sp_, p_drop=p, traj=tr, n=len(rs), rmse_eff_mean_cm=e.mean(),
                         rmse_eff_ci_lo=lo, rmse_eff_ci_hi=hi, success_rate=k / len(rs), succ_ci_lo=sl, succ_ci_hi=sh,
                         tpe10_mean=mean_of("tpe10"), zeta_rms_h2_mean=mean_of("zeta_rms_h2"),
                         dm_err5_mean=mean_of("dm_err@5"), dc_err5_mean=mean_of("dc_err@5"),
                         clip_frac_mean=mean_of("clip_frac")))
    return rows


def hyp_table(res):
    rows = []
    keys = []
    for r in res:
        sp = r["sp"]
        if sp["family"] == "DESC":
            continue
        k = (sp["hyp"], sp["s"], sp["speed"])
        if k not in keys:
            keys.append(k)
    for k in keys:
        by = {r["sp"]["traj"]: r for r in res if (r["sp"]["hyp"], r["sp"]["s"], r["sp"]["speed"]) == k}
        if set(by) != set(TRAJS):
            continue
        c, f = by["circle"], by["fig8"]
        est = c["rej"] and f["rej"]
        dl = ""
        if est and k[0].startswith(("H1a", "H1b")):
            a, b = c["mean"] <= -DELTA, f["mean"] <= -DELTA
            dl = "達實質門檻" if a and b else ("統計上可偵測，但小於事先定的實質門檻" if not a and not b else
                                              f"分軌跡(circle={'達' if a else '未達'},fig8={'達' if b else '未達'})")
        ev = ""
        if not est:
            ev = hyp_evidence(c, f)
        rows.append(dict(hyp=k[0], s=k[1], speed=k[2], circle_rejected=c["rej"], fig8_rejected=f["rej"], established=est,
                         delta_label=dl, evidence_label=ev))
    return rows


def test_rows(res):
    out = []
    for r in res:
        sp = r["sp"]
        out.append(dict(family=sp["family"], test_id=tid(sp), hyp=sp["hyp"], s=sp["s"], speed=sp["speed"], traj=sp["traj"],
                        A=a_name(sp["A"]) if sp["kind"] != "slope_neg" else "Da(s)-Db(s)",
                        B=a_name(sp["B"]) if sp["kind"] != "slope_neg" else "",
                        n=r["n"], mean_diff_cm=r["mean"], ci95_lo=r["ci95"][0], ci95_hi=r["ci95"][1],
                        ci90_lo=r["ci90"][0], ci90_hi=r["ci90"][1], rel_effect=r["rel"], p_onesided=r["p"],
                        p_holm_adj=r["p_adj"], rejected=r["rej"] if sp["family"] != "DESC" else "", delta_layer=delta_layer(r)))
    return out


def slopes_rows(res):
    rows = []
    for r in res:
        if r["Y"] is None:
            continue
        sp = r["sp"]
        for i in range(r["n"]):
            d = dict(hyp=sp["hyp"], traj=sp["traj"], seed_idx=i, slope=r["x"][i])
            for j, s in enumerate(S_LIST):
                d[f"y_s{s:g}"] = r["Y"][i, j]
            rows.append(d)
    return rows


def sensitivity(ctx, specs, allow_partial, res_main):
    rows = []

    def add(variant, res, extra=""):
        for r in res:
            sp = r["sp"]
            rows.append(dict(variant=variant, test_id=tid(sp), hyp=sp["hyp"], s=sp["s"], traj=sp["traj"], n=r["n"],
                             mean_diff_cm=r["mean"], p_onesided=r["p"], p_holm_adj=r["p_adj"], rejected=r["rej"],
                             ci90_lo=r["ci90"][0], ci90_hi=r["ci90"][1], note=extra))
    # (a) P2d rm() 規則重算 F1，並計算受影響集數
    resa, _ = run_tests(ctx, specs, allow_partial, p2d=True, families={"F1"})
    aff = sum(1 for d in ctx.R.values() for r in d.values()
              if (not r.get("success", False)) and r.get("finite", True) and np.isfinite(float(r["rmse"])) and float(r["rmse"]) <= 1.0)
    add("a_p2d_rm_rule", resa, f"受影響集數(全部載入的集中 success=False 且 rmse<=1m)={aff}")
    # (b) H3 ±0.5 cm（不進 Holm；rejected 欄 = 原始 p<0.05 且 90% CI 在界內）
    resb, _ = run_tests(ctx, specs, allow_partial, families={"F3", "F6"}, tost_bound=TOST_B2)
    for r in resb:
        r["p_adj"] = float("nan")
        lo, hi = r["ci90"]
        r["rej"] = bool(r["p"] < ALPHA and lo > -TOST_B2 and hi < TOST_B2)
    add("b_tost_0.5cm", resb, "未校正 p")
    # (c) 排除任何 success=False 的集（成對刪除：任一方失敗的 seed 整對刪），只報
    resc, _ = run_tests(ctx, specs, allow_partial, drop_failed=True, families={"F1"})
    add("c_drop_failed_pairs", resc, "成對刪除")
    return rows


def md_table(headers, rows):
    s = "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def fnum(v, d=2):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:+.{d}f}"


def write_md(path, res, hyps, skipped, args, sens):
    by = {(r["sp"]["hyp"], r["sp"]["s"], r["sp"]["speed"], r["sp"]["traj"]): r for r in res}
    L = ["# PREREG 結果（腳本自動產生，不手改）\n"]
    if args.allow_partial:
        L.append(f"> **DRY-RUN**：輸入 `{args.input}`，seeds {args.seeds}。缺格子已略過，數字不是結論。\n")
    L.append("## 逐 (假設, s) 結論（§6）\n")
    rows = []
    for h in hyps:
        if h["hyp"] in ("H2", "H2-LOCM", "H4", "H3") or h["hyp"].startswith("DROP"):
            continue
        c, f = (by[(h["hyp"], h["s"], h["speed"], t)] for t in TRAJS)
        concl = ("成立" if h["established"] else ("單軌成立" if h["evidence_label"].startswith("單軌") else "不成立"))
        rows.append([h["hyp"], h["s"], f"{fnum(c['mean'])} [{fnum(c['ci95'][0])},{fnum(c['ci95'][1])}] p_adj={c['p_adj']:.4f}",
                     f"{fnum(f['mean'])} [{fnum(f['ci95'][0])},{fnum(f['ci95'][1])}] p_adj={f['p_adj']:.4f}",
                     concl, h["delta_label"], h["evidence_label"]])
    L.append(md_table(["假設", "s", "circle 差(cm)[95%CI]", "fig8 差(cm)[95%CI]", "結論", "Δ層", "證據層"], rows))
    L.append("## H2 / H3 / H4 / 掉包\n")
    rows = []
    for h in hyps:
        if h["hyp"] in ("H2", "H2-LOCM", "H4", "H3") or h["hyp"].startswith("DROP"):
            c, f = (by[(h["hyp"], h["s"], h["speed"], t)] for t in TRAJS)
            rows.append([h["hyp"], h["s"], f"{fnum(c['mean'],3)} p_adj={c['p_adj']:.4f}", f"{fnum(f['mean'],3)} p_adj={f['p_adj']:.4f}",
                         "成立" if h["established"] else "不成立", h["evidence_label"]])
    L.append(md_table(["假設", "s", "circle", "fig8", "結論", "證據層"], rows))
    L.append("## 描述性：B_DOB(s)、總收益 CON-ATT − EKF-L（點估計 [95% CI]，cm）\n")
    rows = []
    for hy in ("B_DOB", "B_LOCM", "TOTAL(CON-EKF-L)"):
        for s in S_LIST:
            cells = []
            for t in TRAJS:
                r = by.get((hy, s, 1.0, t))
                cells.append("n/a" if r is None else f"{fnum(r['mean'])} [{fnum(r['ci95'][0])},{fnum(r['ci95'][1])}]")
            rows.append([hy, s] + cells)
    L.append(md_table(["量", "s", "circle", "fig8"], rows))
    L.append("## 略過的檢定（缺資料）\n")
    L.append("\n".join(f"- {a}：{b}" for a, b in skipped) or "無")
    L.append("\n\n## 敏感度分析（摘要，細節見 sensitivity.csv）\n")
    for v in sorted({s["variant"] for s in sens}):
        g = [s for s in sens if s["variant"] == v]
        L.append(f"- {v}：{len(g)} 列；{g[0]['note']}")
    open(path, "w").write("\n".join(L) + "\n")


def make_figs(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = {(r["sp"]["hyp"], r["sp"]["s"], r["sp"]["traj"]): r for r in res if r["sp"]["speed"] == 1.0}
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for a, t in zip(ax, TRAJS):
        for hy, lab in (("H1a", "Da = GEO - EKF-L"), ("H1b", "Db = CON-ATT - GEO")):
            pts = [(s, by[(hy, s, t)]) for s in S_LIST if (hy, s, t) in by]
            if pts:
                s_ = [p[0] for p in pts]
                m = np.array([p[1]["mean"] for p in pts])
                lo = np.array([p[1]["ci95"][0] for p in pts])
                hi = np.array([p[1]["ci95"][1] for p in pts])
                a.errorbar(s_, m, yerr=[m - lo, hi - m], marker="o", capsize=3, label=lab)
        a.axhline(0, color="gray", lw=0.8)
        a.axhline(-DELTA, color="gray", ls=":", lw=0.8)
        a.set_title(t)
        a.set_xlabel("s")
    ax[0].set_ylabel("cm (A - B)")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_H1_by_s.png"), dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for a, t in zip(ax, TRAJS):
        pts = [(s, by[("B_DOB", s, t)]) for s in S_LIST if ("B_DOB", s, t) in by]
        if pts:
            m = np.array([p[1]["mean"] for p in pts])
            lo = np.array([p[1]["ci95"][0] for p in pts])
            hi = np.array([p[1]["ci95"][1] for p in pts])
            a.errorbar([p[0] for p in pts], m, yerr=[m - lo, hi - m], marker="o", capsize=3)
        a.axhline(0, color="gray", lw=0.8)
        a.set_title(t)
        a.set_xlabel("s")
    ax[0].set_ylabel("B_DOB = DOB - CON-ATT (cm)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_H2_B_DOB.png"), dpi=120)
    plt.close(fig)


def parse_seeds(s):
    a, b = s.split(":")
    return list(range(int(a), int(b)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/prereg/episodes.jsonl")
    ap.add_argument("--out", default="results/prereg")
    ap.add_argument("--seeds", default="3000:3040", help="a:b（不含 b）；正式為 3000:3040")
    ap.add_argument("--allow-partial", action="store_true", help="dry-run 用：缺格子略過並列出（正式分析不得用）")
    args = ap.parse_args(argv)
    seeds = parse_seeds(args.seeds)
    os.makedirs(args.out, exist_ok=True)
    R = load_rows(args.input, set(seeds))
    ctx = Ctx(R, seeds)
    specs = build_specs(seeds)
    res, skipped = run_tests(ctx, specs, args.allow_partial)
    hyps = hyp_table(res)
    sens = sensitivity(ctx, specs, args.allow_partial, res)
    write_csv(f"{args.out}/cells.csv", ["method", "profile", "speed", "p_drop", "traj", "n", "rmse_eff_mean_cm", "rmse_eff_ci_lo",
              "rmse_eff_ci_hi", "success_rate", "succ_ci_lo", "succ_ci_hi", "tpe10_mean", "zeta_rms_h2_mean", "dm_err5_mean",
              "dc_err5_mean", "clip_frac_mean"], cells_table(R))
    write_csv(f"{args.out}/tests.csv", ["family", "test_id", "hyp", "s", "speed", "traj", "A", "B", "n", "mean_diff_cm", "ci95_lo",
              "ci95_hi", "ci90_lo", "ci90_hi", "rel_effect", "p_onesided", "p_holm_adj", "rejected", "delta_layer"], test_rows(res))
    write_csv(f"{args.out}/cells_hyp.csv", ["hyp", "s", "speed", "circle_rejected", "fig8_rejected", "established", "delta_label",
              "evidence_label"], hyps)
    write_csv(f"{args.out}/h2_h4_slopes.csv", ["hyp", "traj", "seed_idx", "slope"] + [f"y_s{s:g}" for s in S_LIST], slopes_rows(res))
    write_csv(f"{args.out}/sensitivity.csv", ["variant", "test_id", "hyp", "s", "traj", "n", "mean_diff_cm", "p_onesided",
              "p_holm_adj", "rejected", "ci90_lo", "ci90_hi", "note"], sens)
    write_md(f"{args.out}/PREREG_results.md", res, hyps, skipped, args, sens)
    make_figs(res, args.out)
    print(f"完成：{len(res)} 個檢定/描述量，略過 {len(skipped)} 個；輸出在 {args.out}")
    for a, b in skipped:
        print("  略過", a, "—", b)


if __name__ == "__main__":
    main()
