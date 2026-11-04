"""PREREG 檢定力模擬（只讀 P2d 已有的 pilot 資料 results/p2d/sweep.jsonl；不跑環境）。輸出 results/prereg/power.md。

資料：results/p2d/sweep.jsonl（seeds 2200-2219，每格 n=20 seeds，2 軌跡 circle/fig8，speed 1.0，hp = best.json），RMSE 取 P2d 的 rm()（>1 m 或發散記 1 m）。
  EKF-L+GEO 只在 s ∈ {0, 1, 2.5} 有資料；s=0.5、1.75 的 GEO 用「同 seed、同軌跡的 GEO RMSE 對 s 線性內插」補（標為內插）。
檢定（對應 docs/PREREG.md §5；n=40）：
  H1a：Da(s) = EKF-L+GEO − EKF-L  < 0（單尾 sign-flip），5 個 s x 2 軌跡
  H1b：Db(s) = CON-ATT − EKF-L+GEO < 0，5 個 s x 2 軌跡     -> 主家族 F1 = 20 個檢定，Holm（用單尾 p）；某 (假設, s) 成立 = 兩軌跡都被拒絕
  H2 ：每 seed 的 OLS 斜率（B_DOB(s) = RMSE(DOB) − RMSE(CON-ATT) 對 s ∈ {0,0.5,1,1.75,2.5}）> 0（單尾 sign-flip），2 軌跡  -> F2（2 個）
  H3 ：D3(s=1) = CON-ATT − CEN-ATT，TOST ±0.25 cm（兩個單尾 sign-flip，p = 較大者），2 軌跡 -> F3（2 個）
  次要：H2 的 LOC-M 斜率（F4，2 個）；H4 = 每 seed 的 (Da − Db) 對 s 的斜率 < 0（F5，2 個）；H3 在其他 s（F6，8 個）。
模擬：從 20 個 seed 有放回抽 n=40（同一組 seed 索引用於所有 s、方法、軌跡，保留相關），算全部檢定，Holm 在各家族內，重複 SIMS 次；power = 通過比例。
  情境：as-observed（直接用觀察到的差）、sd x1.5（偏離均值的部分放大）。另有「真均值平移」表（單一格、保守地用 α = 0.05/20）。
sign-flip：FLIPS 次隨機翻號（固定一張翻號矩陣，所有檢定共用），p = (1 + #{翻號均值 <= 觀察均值}) / (FLIPS + 1)。種子固定。
限制：20 個 seeds 重抽到 40，重複 seed 使有效資訊只有 20；觀察到的均值本身有抽樣誤差，as-observed 的 power 可能偏樂觀；這些 seeds 已被看過。
"""
import os
import sys

import numpy as np

sys.path.insert(0, "experiments")
from p2d_run import SWEEP_SEEDS, TRAJS, table

OUT = "results/prereg"
N, SIMS, FLIPS, DELTA, DELTA_SUB = 40, 2000, 5000, 0.25, 0.5
S = [0, 0.5, 1, 1.75, 2.5]
GEO_S = [0, 1, 2.5]
SG = np.random.default_rng(12345).choice([-1.0, 1.0], size=(FLIPS, N))


def p_less(x):
    """單尾 sign-flip：H1: mean(x) < 0。"""
    return (1 + int(((SG @ x) / len(x) <= x.mean()).sum())) / (FLIPS + 1)


def p_tost(x, delta):
    return max(p_less(x - delta), p_less(-(x + delta)))


def holm(ps, alpha=0.05):
    """回傳與 ps 同序的 bool（Holm step-down，用單尾 p）。"""
    ps = np.asarray(ps)
    order = np.argsort(ps)
    rej = np.zeros(len(ps), bool)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - rank):
            rej[i] = True
        else:
            break
    return rej


def load_arrays():
    D = table("results/p2d/sweep.jsonl")

    def R(m, s):   # (20 seeds, 2 軌跡) cm
        return np.array([[D[(m, f"s={s:g}", t, "")][sd][0] * 100 for t in TRAJS] for sd in SWEEP_SEEDS])

    def geo(s):
        if s in GEO_S:
            return R("EKF-L+GEO", s)
        lo = max(g for g in GEO_S if g < s); hi = min(g for g in GEO_S if g > s)
        w = (s - lo) / (hi - lo)
        return (1 - w) * R("EKF-L+GEO", lo) + w * R("EKF-L+GEO", hi)

    A = {}
    for s in S:
        A[("H1a", s)] = geo(s) - R("EKF-L", s)
        A[("H1b", s)] = R("CON-ATT", s) - geo(s)
        A[("H3", s)] = R("CON-ATT", s) - R("CEN-ATT", s)
    xs = np.array(S, float) - np.mean(S)
    def slope(M):    # M: (seeds, 2, len(S)) -> (seeds, 2) OLS 斜率，x = s 的數值
        return (M * xs).sum(-1) / (xs ** 2).sum()
    for nm, L in (("H2-DOB", "DOB"), ("H2-LOC-M", "LOC-M")):
        A[(nm, "slope")] = slope(np.stack([R(L, s) - R("CON-ATT", s) for s in S], -1))
    A[("H4", "slope")] = slope(np.stack([A[("H1a", s)] - A[("H1b", s)] for s in S], -1))
    for s in S:
        A[("B-DOB", s)] = R("DOB", s) - R("CON-ATT", s)
        A[("B-LOC-M", s)] = R("LOC-M", s) - R("CON-ATT", s)
    return A


def reshape(d, mean=None, k=1.0):
    d = np.asarray(d, float)
    mu = d.mean(0)
    d = mu + (d - mu) * k
    if mean is not None:
        d = d - d.mean(0) + np.asarray(mean, float)
    return d


def tests(A):
    """(家族, 假設, s, 檢定函式, 型別) 列表；每個檢定對 2 軌跡各做一次。"""
    T = []
    for s in S:
        T.append(("F1", "H1a", s))
        T.append(("F1", "H1b", s))
    for nm in ("H2-DOB",):
        T.append(("F2", nm, "slope"))
    T.append(("F3", "H3", 1))
    T.append(("F4", "H2-LOC-M", "slope"))
    T.append(("F5", "H4", "slope"))
    for s in S:
        if s != 1:
            T.append(("F6", "H3", s))
    return T


def pval(h, x):
    if h == "H3":
        return p_tost(x, DELTA)
    if h == "H2-DOB" or h == "H2-LOC-M":
        return p_less(-x)            # 斜率 > 0
    return p_less(x)                 # H1a、H1b、H4：< 0


def simulate(A, k=1.0, rng=None):
    T = tests(A)
    B = {key: reshape(v, None, k) for key, v in A.items()}
    names = [(f, h, s) for f, h, s in T]
    cnt = {n: np.zeros(2) for n in names}      # 各軌跡被 Holm 拒絕
    both = {n: 0 for n in names}
    sub = {n: 0 for n in names}                # 兩軌跡都拒絕且兩軌跡點估計 ≤ −Δsub（只對 H1a/H1b）
    est_ok = {n: 0 for n in names}             # 兩軌跡點估計 ≤ −Δsub（不看顯著性）
    fams = sorted({f for f, _, _ in T})
    for _ in range(SIMS):
        idx = rng.integers(0, 20, N)
        P = {}
        for f, h, s in T:
            X = B[(h, s)][idx]
            P[(f, h, s)] = np.array([pval(h, X[:, j]) for j in range(2)])
        for f in fams:
            ks = [n for n in names if n[0] == f]
            ps = np.concatenate([P[n] for n in ks])
            rej = holm(ps).reshape(len(ks), 2)
            for n, r in zip(ks, rej):
                cnt[n] += r
                both[n] += bool(r.all())
                if n[1] in ("H1a", "H1b"):
                    m = B[(n[1], n[2])][idx].mean(0)
                    ok = bool((m <= -DELTA_SUB).all())
                    est_ok[n] += ok
                    sub[n] += bool(r.all()) and ok
    return {n: dict(circle=cnt[n][0] / SIMS, fig8=cnt[n][1] / SIMS, both=both[n] / SIMS, est=est_ok[n] / SIMS, sub=sub[n] / SIMS) for n in names}


def shift_table(A, rng, alpha):
    """單一格（兩軌都過）的 power：真均值平移，sd 同觀察；α 用 Holm 首步 alpha（保守）。"""
    rows = []
    for h in ("H1a", "H1b"):
        for s in S:
            row = [h, s]
            for m in (-0.1, -0.2, -0.3, -0.5):
                d = reshape(A[(h, s)], [m, m])
                hit = np.zeros((SIMS, 2), bool)
                for r in range(SIMS):
                    idx = rng.integers(0, 20, N)
                    hit[r] = [p_less(d[idx, j]) < alpha for j in range(2)]
                row.append(hit.all(1).mean())
            rows.append(row)
    return rows


def main():
    rng = np.random.default_rng(0)
    A = load_arrays()
    L = ["# PREREG 檢定力模擬（腳本 experiments/prereg_power.py 產出）\n",
         f"n = {N} seeds/格；模擬 {SIMS} 次；sign-flip {FLIPS} 次；種子固定。資料：`results/p2d/sweep.jsonl`（seeds 2200–2219，n=20/格，2 軌跡，hp=best.json），"
         "有放回重抽到 n=40，同一組 seed 索引貫穿所有 s、方法、軌跡。這些 seeds 已被看過。"
         "EKF-L+GEO 只有 s ∈ {0, 1, 2.5}；s=0.5、1.75 用同 seed 同軌跡的 GEO RMSE 對 s 線性內插（**內插**，標 †；延遲是階梯、pos_std 在 s>1 飽和，內插只是粗估）。\n",
         "限制：20 個 seed 重抽到 40，有效資訊仍只有 20 個 seed；觀察到的均值本身有抽樣誤差，as-observed 的 power 可能偏樂觀；2200–2219 已被看過；"
         "sd 在各 s 的估計各只有 20 個點。檢定力是規劃用，不是證據。\n",
         f"Holm 家族（各 α=0.05，用單尾 p）：F1 = H1a、H1b 各 5 s x 2 軌跡（20 個）；F2 = H2(DOB) x 2；F3 = H3(s=1, TOST ±{DELTA}) x 2；"
         "次要：F4 = H2(LOC-M) x 2；F5 = H4(來源交叉斜率) x 2；F6 = H3 在其他 s x 8。某 (假設, s) 成立 = 兩條軌跡都被該家族 Holm 拒絕。\n"]
    L.append("## 觀察到的配對差（cm；負 = 前者較好；n=20）\n\n| 比較 | s | circle 平均 (sd) | fig8 平均 (sd) |\n|---|---|---|---|")
    for h, lab in (("H1a", "Da = GEO − EKF-L"), ("H1b", "Db = CON − GEO"), ("H3", "D3 = CON − CEN")):
        for s in S:
            d = A[(h, s)]
            dag = "†" if h != "H3" and s not in GEO_S else ""
            L.append(f"| {lab} | {s:g}{dag} | {d[:,0].mean():+.3f} ({d[:,0].std(ddof=1):.3f}) | {d[:,1].mean():+.3f} ({d[:,1].std(ddof=1):.3f}) |")
    for h, lab in (("H2-DOB", "H2 斜率 (DOB)"), ("H2-LOC-M", "H2 斜率 (LOC-M)"), ("H4", "H4 斜率 (Da−Db)")):
        d = A[(h, "slope")]
        L.append(f"| {lab}（cm / 單位 s） | 全 s | {d[:,0].mean():+.3f} ({d[:,0].std(ddof=1):.3f}) | {d[:,1].mean():+.3f} ({d[:,1].std(ddof=1):.3f}) |")
    out = {}
    for nm, k in (("as-observed", 1.0), ("sd x1.5", 1.5)):
        out[nm] = simulate(A, k, np.random.default_rng(1))
    for nm, R_ in out.items():
        L.append(f"\n## 檢定力：{nm}（Holm 在家族內；欄 `兩軌都過` 為該 (假設, s) 成立的機率）\n")
        L.append("| 家族 | 假設 | s | circle | fig8 | **兩軌都過** | 兩軌點估計 ≤ −0.5 cm | 兩軌都過且 ≤ −0.5 cm |\n|---|---|---|---|---|---|---|---|")
        for (f, h, s), v in R_.items():
            dag = "†" if h in ("H1a", "H1b") and s not in GEO_S else ""
            est = f"{v['est']:.3f}" if h in ("H1a", "H1b") else "—"
            sub = f"{v['sub']:.3f}" if h in ("H1a", "H1b") else "—"
            L.append(f"| {f} | {h} | {s}{dag} | {v['circle']:.3f} | {v['fig8']:.3f} | **{v['both']:.3f}** | {est} | {sub} |")
    alpha = 0.05 / 20
    L.append(f"\n## H1a / H1b：真均值平移（sd 同觀察；單格兩軌都過；α = 0.05/20 = {alpha:.4f}，F1 的 Holm 首步，為保守下界）\n")
    L.append("| 假設 | s | 真均值 −0.1 | −0.2 | −0.3 | −0.5 |\n|---|---|---|---|---|---|")
    for h, s, *p in shift_table(A, np.random.default_rng(2), alpha):
        dag = "†" if s not in GEO_S else ""
        L.append(f"| {h} | {s:g}{dag} | " + " | ".join(f"{x:.3f}" for x in p) + " |")
    L.append("\n（H3 的 TOST 邊界型一錯誤檢查沒有在本腳本算：上一版 ±0.25 平移實驗顯示真均值 = ±0.25 時 power ≈ 0。）")
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/power.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
