"""prereg_analyze.py 的合成資料測試（不讀真實結果）。"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import prereg_analyze as pa  # noqa: E402


def test_signflip_null_roughly_uniform():
    rng = np.random.default_rng(123)
    ps = np.array([pa.signflip_p(rng.normal(0, 1, 20)) for _ in range(400)])
    assert abs(ps.mean() - 0.5) < 0.06
    assert 0.02 < (ps < 0.05).mean() < 0.09          # 型一錯誤率約 5%
    assert 0.18 < (ps < 0.25).mean() < 0.32
    assert 0.40 < (ps < 0.5).mean() < 0.60


def test_signflip_toy_example():
    p = pa.signflip_p(np.array([-0.6, -0.3, -0.7, -0.2]))
    assert abs(p - 1 / 16) < 0.004                    # PREREG §10.4
    assert pa.signflip_p(np.full(40, -1.0)) < 1e-4    # 全負：下限 ~ 1/B
    assert pa.signflip_p(np.full(40, 1.0)) > 0.99


def test_holm_doc_examples():
    adj, rej = pa.holm([0.001, 0.02, 0.04])
    assert rej.all()
    adj, rej = pa.holm([0.001, 0.03, 0.04])
    assert list(rej) == [True, False, False]
    np.testing.assert_allclose(adj, [0.003, 0.06, 0.06])
    # 順序不影響；單調性；上限 1
    adj, rej = pa.holm([0.04, 0.001, 0.03])
    np.testing.assert_allclose(adj, [0.06, 0.003, 0.06])
    assert pa.holm([0.9, 0.8])[0].max() <= 1.0
    # 20 個檢定的第一步門檻 0.0025
    _, rej = pa.holm([0.0625] + [0.5] * 19)
    assert not rej.any()
    _, rej = pa.holm([0.002] + [0.5] * 19)
    assert rej[0]


def test_tost_known_cases():
    rng = np.random.default_rng(1)
    x = rng.normal(-0.03, 0.08, 40)
    p, p1, p2, lo, hi = pa.tost(x, 0.25)
    assert p < 0.001 and -0.25 < lo < hi < 0.25      # 等價
    x2 = rng.normal(0.30, 0.08, 40)                  # 均值超出界限
    assert pa.tost(x2, 0.25)[0] > 0.5
    x3 = np.tile([-1.0, 1.0], 20)                    # 變異太大：測不出等價
    assert pa.tost(x3, 0.25)[0] > 0.05
    # p_TOST 是兩個單尾中較大者
    p, p1, p2, _, _ = pa.tost(rng.normal(0.2, 0.1, 40), 0.25)
    assert p == max(p1, p2)


def test_rmse_eff_truncation_rule():
    ok = dict(rmse=0.0286, success=True, finite=True)
    assert pa.rmse_eff(ok) == pytest.approx(2.86)
    assert pa.rmse_eff(dict(rmse=0.30, success=False, finite=True)) == 100.0      # §10.4：失敗記 1 m
    assert pa.rmse_eff(dict(rmse=0.30, success=False, finite=True), p2d_rule=True) == pytest.approx(30.0)
    assert pa.rmse_eff(dict(rmse=1.5, success=True, finite=True)) == 100.0
    assert pa.rmse_eff(dict(rmse=float("inf"), success=False, finite=False)) == 100.0
    assert pa.rmse_eff(dict(rmse=float("nan"), success=True, finite=True)) == 100.0
    assert pa.rmse_eff(dict(rmse=None, success=True, finite=True)) == 100.0
    assert pa.rmse_eff(dict(rmse=0.05, success=True, finite=False)) == 100.0
    assert pa.rmse_eff(dict(rmse=0.5, success=True, finite=True, ), p2d_rule=True) == pytest.approx(50.0)


def test_slope_example():
    sl = pa.ols_slope(np.array([[0.3, 0.5, 1.0, 2.0, 3.0]]))
    assert sl[0] == pytest.approx(4.43 / 3.95, abs=0.01)   # 1.12


def test_bootstrap_ci_contains_mean_and_is_shared():
    x = np.random.default_rng(5).normal(-1, 1, 40)
    lo, hi = pa.boot_ci(x, 95)
    l90, h90 = pa.boot_ci(x, 90)
    assert lo < x.mean() < hi and l90 > lo and h90 < hi
    assert pa.boot_idx(40) is pa.boot_idx(40)


def test_hp_matching_and_strict_missing():
    assert pa.hp_matches("CON-ATT", 1.0, {"lam": 0.9998, "prior": 2.0})              # pilot 沒記 iters
    assert pa.hp_matches("CON-ATT", 1.0, {"lam": 0.9998, "prior": 2.0, "iters": 2, "p_drop": 0.5})
    assert not pa.hp_matches("DOB", 1.0, {"fc": 8.0})
    assert pa.hp_matches("HYB", 1.5, {"fc": 1000.0, "tau": 0.07})
    ctx = pa.Ctx({}, [3000, 3001])
    with pytest.raises(SystemExit):
        pa.run_tests(ctx, pa.build_specs([3000, 3001]), allow_partial=False)
    res, skipped = pa.run_tests(ctx, pa.build_specs([3000, 3001]), allow_partial=True)
    assert res == [] and len(skipped) > 0


def test_end_to_end_synthetic_f1_detects_effect():
    seeds = list(range(3000, 3012))
    rng = np.random.default_rng(0)
    R = {}
    base = {"EKF-L": 3.0, "EKF-L+GEO": 2.0, "CON-ATT": 1.5}   # cm：真有效應
    for m, mu in base.items():
        for s in pa.S_LIST:
            for tr in pa.TRAJS:
                R[(m, f"s={s:g}", 1.0, 0.0, tr)] = {
                    sd: dict(rmse=(mu + rng.normal(0, 0.1)) / 100, success=True, finite=True) for sd in seeds}
    ctx = pa.Ctx(R, seeds)
    res, sk = pa.run_tests(ctx, pa.build_specs(seeds), True, families={"F1"})
    assert len(res) == 20
    assert all(r["rej"] for r in res)
    assert all(r["sp"]["hyp"] in ("H1a", "H1b") and r["mean"] < 0 for r in res)
