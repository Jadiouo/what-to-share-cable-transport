# NUMBERS: where each number in main.tex (v3) comes from
Paths relative to repo root. Applies to `paper/workshop/main.tex` (v3, 7 pages). The previous version (v2) is described in the section "What changed in v3".
Abbreviations: "tests" = results/prereg/tests.csv (family F1 unless noted; columns mean_diff_cm, ci95_*, p_holm_adj, rel_effect). "sens" = results/prereg/sensitivity.csv. "cells" = results/prereg/cells.csv. "episodes" = results/prereg/episodes.jsonl. "RECOMPUTED" = not in any result file; computed from episodes by a script in `paper/workshop/scripts/` (listed at the bottom). All scripts run from the repo root, single-threaded, in under a minute.

## Design / protocol (text, not results)
- Freeze at commit 5dd3cf7 (protocol v3), seeds 3000-3039, 3240 episodes, no deviations as defined in the preregistration: docs/PREREG.md (header and section 2); episode count and seed range checked by the loader line printed by any of the scripts below ("3240 episodes").
- N=3, 1 kg quad, payload 1 kg, r=0.15 m, rope 1 m, dt 0.01: src/smpc/envs/params.py, src/smpc/eval/episode.py (M_NOM, PAYLOAD_RADIUS), multi_payload.py default rope_length=1.0.
- dm in [-0.3,0.4], |dc|<=0.04: src/smpc/eval/episode_est.py DM_RANGE, DC_MAX.
- MPPI K=512, H=20, dt_plan 0.05: configs/mppi_frozen.json.
- circle R=0.8, w=1.2; fig8 A=0.8,B=0.4,w=0.9; 10 s: src/smpc/control/trajectories.py, eval/episode.py DURATION.
- s scaling (sigma_a 0.15+0.45s, bias 0.1s, kerr 0.05s, delay 0/2/3 at s<0.5/<1.75/else, pos_std 0.01 min(s,1), rope-direction noise 0.005+0.005s): src/smpc/eval/sensing.py (PROFILES, scaled_profile). Appendix A, Table III.
- Method x information table (Table I), method descriptions: src/smpc/adapt/ekf_geo.py header; docs/PREREG.md 3.1; ekf_local.py; con_att.py (HCMCI, iters=2, path graph, relay N-1, max_age=5, dropout rule). use_dc=0 for EKF-L+GEO: results/p2d/best.json (EKF-L+GEO hp); ekf_geo.py DEFAULTS has use_dc=1 but the frozen hp overrides it. CON-ATT estimates dc (state [dm, dc, zeta_1..N]).
- RMSE_eff definition, success definition (last-second mean error < 0.10 m and max error < 0.5 m), 1 m rule: docs/PREREG.md section 4; src/smpc/eval/episode_p2c.py (final_err, err.max()).
- sign-flip 1e5, bootstrap 1e4, Holm families F1-F8, TOST +-0.25 (+-0.5 sensitivity), Delta=0.5 cm: docs/PREREG.md sections 1, 5; experiments/prereg_analyze.py (B_PERM, B_BOOT).
- Tuning seeds 2020-2024, s=1, speed 1.0; budget 14 (EKF-L) / 10 (EKF-L+GEO) / 10 (LOC-M) / 9 (CEN-ATT, shared by CON-ATT) / 8 (DOB), 10 episodes per point (5 seeds x 2 trajectories); EKF-L extension = recorded deviation from the tuning specification: docs/PREREG.md 3.3. Boundary optima (LOC-M lam, CEN-ATT prior): results/p2d/best.json `still_boundary`.
- Pilot seeds 2200-2219 (also 2020-2029, 2100-2119) were seen before the freeze; failure-scoring rule set after the pilot: docs/PREREG.md section 2 (last bullets) and section 4 (bullet on differences from the pilot).
- Pilot power about 0.1 for H1b at s>=1.75: docs/PREREG.md section 5 and results/prereg/power.md.
- Team size: attitude-aware model breaks down at N=4 and 6 (preliminary smoke runs, flights completed): docs/PREREG.md section 8 (known limitations). No load cell: same section.
- How MPPI uses the estimates (theta_hat_i into team.set_theta, eta=1+zeta_hat) and thrust division by 1+zeta_hat: src/smpc/eval/episode_p2c.py lines ~138-172.
- Relay N-1 rounds, drop applies to relay and consensus rounds, stale packets (>5 steps) fall back to nominal geometry: src/smpc/adapt/con_att.py docstring.

## Fig. 1 (simulator scene)
- figs/sim_snapshot_t4.png: paper/workshop/scripts/render_snapshot.py (reruns CON-ATT, s=1, circle, seed 3000 with frozen hp; RMSE 0.025769, identical to episodes; render on CPU llvmpipe, frame t=4 s, az 60, el -16, dist 2.8).

## Fig. 2 (ladder of methods)
- TikZ schematic in main.tex; no data.

## Abstract / Introduction / Conclusion
- "2.3-3.7 cm" (CON-ATT tracking error, 10 cells at speed 1.0, no packet drop): RECOMPUTED, scripts/recompute_v3_numbers.py block (2): min 2.347 (circle s=0), max 3.713 (fig8 s=2.5); equals cells rmse_eff_mean_cm.
- "about 11-14%" (H1a effect relative to EKF-L's error, s>=0.5, 7 cells without fig8 s=2.5): tests rel_effect (H1a, s>=0.5), also RECOMPUTED in recompute_v3_numbers.py block (1): 11.48% (circle s=1) to 14.38% (circle s=2.5).
- "about 0.2-0.65 cm" (range of detectable effects): tests mean_diff_cm, smallest detectable |H1b| 0.160 (circle s=1), largest |H1a| excluding the artifact 0.649.
- "0.25 cm band", "same order as the gains": docs/PREREG.md H3; tests F3.

## Table II / H1 text (Sec. V-A) and Fig. 3
- Table II means: tests F1 mean_diff_cm (rounded 2 dp), all 20 rows. Italic = Holm-adjusted p >= 0.05 (H1a s=0 both; H1b s=1.75, 2.5 both).
- Table II medians: RECOMPUTED (paired differences of RMSE_eff per seed, np.median), scripts/recompute_medians_slopes.py.
- "-0.38 to -0.65" (H1a, s>=0.5, excluding the dagger entry): tests (min -0.3781 circle s=0.5; max -0.6490 circle s=2.5).
- "-2.97" fig8 s=2.5 Da: tests (H1a|s=2.5|fig8). "-0.67 (90% CI -0.96, -0.38)": sens a_p2d_rm_rule, H1a|s=2.5|fig8 (mean -0.665202, ci90 -0.96229, -0.377384). Total -3.20 and H4 fig8 -1.07: results/prereg/PREREG_results.md (TOTAL row; H4 row) / tests F5. "Other 19 entries unchanged": sens(a) mean_diff_cm diffed against tests (only 1 differs).
- "Reaches Delta only for s>=1.75" (H1a) and "-0.503, -0.513": tests F1 and delta_layer; "medians -0.20 to -0.45 for s>=0.5": RECOMPUTED (recompute_medians_slopes.py).
- H1b: "-0.16 to -0.64" for s<=1: tests; "only s=0 reaches Delta; medians -0.27 and -0.14": tests delta_layer; medians RECOMPUTED. "Power about 0.1" for s>=1.75: results/prereg/power.md.
- "H1b adjusted p 0.07 to 0.21" at s=1.75, 2.5: tests p_holm_adj (0.1872, 0.0707, 0.2050, 0.1571).
- "H1a adjusted p 0.001 to 0.033, four between 0.020 and 0.033" (Appendix E) and "five main-family adjusted p between 0.02 and 0.033" (Limitations): tests p_holm_adj (H1a s=1: .0329, .0217; H1b s=1: .0304; H1a s=1.75: .0199, .0199; the last two round to 0.020).
- H4 slopes -0.40, -1.07, adjusted p 0.0004, 0.0029: tests F5 (p_holm_adj 0.00038, 0.0029). H4 without s=0 and with raw-error scoring (-0.16, -0.28, -0.096; circle with raw scoring is unchanged at -0.40): RECOMPUTED, scripts/recompute_h4_sens.py. Circle without s=0 is -0.16493.
- Seed 3024, EKF-L, s=2.5, fig8: rmse 0.0792 m, success False, episodes key prereg|EKF-L|s=2.5|fig8|1.0|3024|q_b=0.0005,r_scale=10.
- Fig. 3 (H1 figure): scripts/make_figs_v3.py -> figs/fig_H1_v3.pdf, formal values from tests F1 (mean_diff_cm, ci95_lo, ci95_hi); the fig8 s=2.5 point and the lower CI end are clipped at -1.4 by the plotting code (display only).

## Sec. V-B, Appendix D (packet loss) and Appendix E (H3)
- H3: -0.021, -0.033 at s=1, adjusted p<1e-4: tests F3 (p_holm_adj 1.99998e-05). 90% CIs [-0.032,-0.010], [-0.043,-0.023]: tests ci90 columns, F3 rows. Means -0.049..-0.005 over five s: tests F3+F6 (min -0.0491 fig8 s=2.5; max -0.0046 circle s=0). +-0.5 band passes: sens b_tost_0.5cm. "CON-ATT slightly better at all five levels": sign of the F3 means.
- p=0.5 vs EKF-L: -0.58, -0.64, adjusted p 1.4e-4, 8e-5: tests F8 (-0.577413, -0.640494; p_holm_adj 0.00013999, 7.99992e-05). Tracking error 2.76/2.83 (p=0.5) vs 2.80/2.85 (p=0), 40 seeds: cells (CON-ATT, s=1, p_drop 0.5 and 0).
- p=0.9 (changed in v3, see below): Table IV and text, same 20 seeds (3000-3019) for the lossless baseline: RECOMPUTED, scripts/recompute_v3_numbers.py block (5) from episodes. Tracking error lossless 2.88/2.91 vs p=0.9 2.80/2.90 cm; zeta RMS 0.0138/0.0140 -> 0.0195/0.0194 (+41%/+39%); dm error at t=5 s 0.0446/0.0476 -> 0.0529/0.0542 (+19%/+14%); 0.5 s tension-prediction error 0.238/0.241 -> 0.292/0.292 (+23%/+22%) (fields zeta_rms_h2, dm_err@5, tpe10 of episodes). The p=0.9 cells in cells.csv (n=20) agree; cells.csv holds only the 40-seed lossless rows, which is why the 20-seed baseline is recomputed.
- p=0.9 vs EKF-L+GEO (same 20 seeds, descriptive, no Holm): tests DESC rows DROP0.9-vs-GEO|s=1: circle -0.250669 (95% CI -0.478698, -0.0481378), fig8 -0.213278 (-0.406646, -0.0417068); also in recompute_v3_numbers.py block (5b).
- Message-layer completeness 99.7% (p=0.5) and 39% (p=0.9): post hoc, new in v3. scripts/recompute_posthoc_extras.py block (b); it drives the real smpc.adapt.con_att.ConATT message layer with synthetic inputs (seeds 3000-3039, 1000 steps; no result file is read). Means over seeds 0.9970 and 0.3940 (min-max 0.9933-0.9993 and 0.3550-0.4540). The script checks that the comms-only variant equals the full ConATT.
- Speed 1.5: H1a@1.5 circle +0.18 (n.s.), fig8 -0.22; H1b@1.5 -0.37, -0.18: tests F7.
- Overconfidence pilot remark: docs/PREREG.md section 2.

## Sec. V-C, Appendix C, Fig. 4 (H2) and failed episodes
- H2 slopes +11.0, +10.6, adjusted p<1e-4: tests F2 (11.0298, 10.5674; p_holm_adj 1.99998e-05). "All 40 seeds slope>0": h2_h4_slopes.csv (RMSE_eff and raw both 40/40; raw RECOMPUTED).
- Raw slopes +1.84 (circle), +1.90 (fig8): RECOMPUTED (recompute_medians_slopes.py). Raw DOB - CON-ATT means 0.40/0.22 at s=0 and 4.95/4.92 at s=2.5: RECOMPUTED. Fig. 4: scripts/make_figs_v3.py (second half) -> figs/fig_H2_raw_v3.pdf; bootstrap CI generated there (not preregistered).
- "31.8 and 27.2 cm" (formal DOB - CON-ATT at s=2.5, Fig. 4 caption): tests DESC rows (B_DOB, s=2.5: 31.8454, 27.2274); RECOMPUTED in recompute_v3_numbers.py block (3).
- 22/80 DOB failures at s=2.5, 58/80 success: cells success_rate (0.70, 0.75 x 40); RECOMPUTED in recompute_v3_numbers.py block (4), which also gives DOB failures per s at speed 1.0 (0, 0, 0, 3, 22 of 80 for s=0..2.5). Wilson 0.70 [0.546,0.819], 0.75 [0.598,0.858]: cells succ_ci_lo/hi. CON-ATT 80/80: cells.
- LOC-M slope +1.34, +2.89: tests F4. LOC-M fails 4 episodes at s=2.5 (1 circle + 3 fig8): cells success_rate (0.975, 0.925 x 40) and recompute_v3_numbers.py block (4) (LOC-M 4 in total).
- Failed-episode set (new in v3): 35 of 3240 failed (DOB 30, LOC-M 4, EKF-L 1), none with a non-finite value, last-second mean error of the failed episodes 10.08-20.26 cm (paper: 10.1 to 20.3), largest maximum error 22.68 cm (paper: no more than 23 cm): RECOMPUTED, recompute_v3_numbers.py block (4) from episodes (fields success, finite, final_err, max_err). Of the 30 DOB failures, 25 are at speed 1.0 and 5 at other speeds (the speed-1.5 runs).

## H3 / Limitations
- Pilot Da(s=1) -0.52/-0.50 vs formal -0.38/-0.42: results/prereg/power.md (-0.520, -0.496) and tests.
- ORACLE and HYB are in the 3240 episodes and are descriptive only (preregistration G5).
- Related-work citations in refs.bib are complete entries, checked against Crossref/arXiv on 2026-11-01.

## Post hoc analyses added in v3
- ORACLE reference (Appendix E, "For scale"): scripts/recompute_posthoc_extras.py block (a), from episodes (RMSE_eff rule), cross-checked against cells rmse_eff_mean_cm (|diff| < 1e-5). At s=1: ORACLE 2.14549/2.37748, EKF-L 3.34206/3.47528, CON-ATT 2.79814/2.85178 cm (circle/fig8). EKF-L minus ORACLE 1.19657/1.09780; EKF-L minus CON-ATT 0.54392/0.62350; ratio 0.45/0.57 ("about half"; the text quotes 0.54 of 1.20 and 0.62 of 1.10).
- Message-layer completeness 99.7% / 39%: see the packet-loss section above (recompute_posthoc_extras.py block (b)).
- Two-sided p-values (Appendix B): scripts/recompute_posthoc_extras.py block (c). It first recomputes the 20 one-sided F1 p-values and Holm values with the frozen functions and stops if they differ from tests (all 20 agree to <5e-6), then recomputes two-sided p-values with the same sign-flip matrix and Holm. Result: 2 of 20 tests lose rejection, H1a s=1 circle (adjusted p 0.060479) and H1b s=1 circle (0.059519); both quoted as 0.060. Supported on both trajectories under two-sided tests: H1a at s=0.5, 1.75, 2.5; H1b at s=0, 0.5 (one-sided: 7 of 10 (hypothesis, s) pairs; two-sided: 5 of 10).

## Added in the fifth batch (reviewer suggestions; all post hoc or descriptive, none preregistered as numbers)
All recomputed by scripts/recompute_posthoc_extras.py blocks (d), (e), (f); output in .scratch/audit_20261118/posthoc_extras_out.txt (blocks (a)-(c) unchanged).
- H4 with raw-RMSE scoring (Appendix C). Block (d) reuses Ctx.slope_Y, ols_slope, the shared sign matrix and signflip_p from experiments/prereg_analyze.py, with run_tests(p2d=True) (the sensitivity-(a) scoring rule: a failed episode with finite RMSE <= 1 m keeps its raw RMSE; non-finite or > 1 m stays 100 cm). With the formal scoring it first reproduces the F5 rows of results/prereg/tests.csv (slope -0.4010 / -1.0695, p_onesided 0.00019 / 0.00292, Holm 0.00038 / 0.00292; |diff| < 5e-6). Raw scoring, figure-eight: slope -0.2828 (95% CI -0.481 to -0.082), one-sided p = 0.003850, Holm within F5 (2 tests) = 0.003850; paper quotes -0.28 and p = 0.0039. The circle is unchanged by the scoring (-0.4010). The paper does not cite the circle or the s>=0.5 p-values. Without s=0: circle -0.1649 (quoted -0.16), figure-eight raw -0.0961 (quoted -0.096), figure-eight formal -1.1609. Family handling: no Holm across families; the F5 family is the two H4 tests, as in the formal run.
- Message sizes (Appendix A; counted from code, not measured traffic). Block (e) instantiates the real classes and reads their shapes. EKF-L+GEO: packet = 6 numbers per source (src/smpc/adapt/ekf_geo.py L63 `self.data = np.zeros((N, N, 6))`, L82 `self.data[i, i] = np.concatenate([pq[i], q[i]])`), relay_rounds = N-1 = 2 (L47), no consensus. CON-ATT: packet = 13 numbers per source (src/smpc/adapt/con_att.py L64 `self.data = np.zeros((N, N, 13))`, L134 `np.concatenate([pq[i], q[i], y[i], f[i], [float(active[i])]])`), relay_rounds = N-1 = 2 (L51), iters = 2 (L45 default; formal hp iters=2). Consensus (L146-151): each iteration mixes Lp (d x d), xp (d), U (d x d), u (d) with d = 3+N = 6 (L52), i.e. two information pairs. Counting: symmetric 6x6 Lambda has 21 upper-triangle entries (checked: max |Lambda - Lambda^T| = 5e-15 after 20 steps with random inputs), so a pair is 21+6 = 27 numbers and a message 54; with full matrices 36+6 = 42 and 84. This matches the header comment, con_att.py L16 "每則訊息 = 兩個 (Λ,ξ) 對（各 21+6 個獨立數）＋ 一個量測封包（13 個數）". Per step per directed link: 2 iterations x 54 = 108 numbers (168 full), not quoted in the paper. A relay round in the code reads a neighbor's whole table (N sources): 18 numbers (GEO), 39 (CON-ATT); this is how the code works, and a stricter implementation could send only new entries. The paper says nothing about bytes, rates or bandwidth.
- Clipped-estimate share at p=0.9 (Appendix D, Table IV last row). clip_frac is NOT thrust saturation: src/smpc/eval/episode_p2c.py L154-155 `th_c = np.clip(theta_now[i], *TH_CLIP); clipn += int(np.any(th_c != theta_now[i]))`, L198 `clip_frac=clipn / max(1, steps * n)`; TH_CLIP = ([-0.7,-0.15,-0.15], [1.0,0.15,0.15]) in src/smpc/eval/episode_est.py L26 (dm in kg, dc_x, dc_y in m). It is the share of (step, quadrotor) pairs whose parameter estimate left its box. CON-ATT, s=1, speed 1.0: p=0.9 vs lossless on seeds 3000-3019: circle 0.00138 -> 0.00310 (+124%), figure-eight 0.00137 -> 0.00320 (+134%); the p=0.9 means equal cells.csv clip_frac_mean (|diff| < 1e-18). p=0.5 vs lossless on seeds 3000-3039: circle 0.00134 -> 0.00131, figure-eight 0.00133 -> 0.00133 (cells.csv agrees, |diff| < 4e-9). The same-seed tracking errors reproduce Table IV (2.880/2.915 vs 2.803/2.904). Seeds with any clipped step: 11/20 -> 18/20 at p=0.9, 18/40 -> 27/40 at p=0.5 (not quoted). Preregistration text required: docs/PREREG.md L71 ("p=0.9 列為描述性：…ζ／δm 誤差與 clip 比例反而變差、RMSE 卻變好，推論為 HCMCI 在高掉包下過度自信…不做假設檢定"), L194 ("p=0.9 只在描述性段落，附 ζ̂／clip 指標，並註明過度自信"), L143 (clip_frac listed as an auxiliary secondary quantity). Overconfidence mechanism as stated in src/smpc/adapt/con_att.py L27-30 (skipped message: weight returns to the receiver, W no longer doubly stochastic, N*[.] counts an agent's own new information up to N times; header marks it "推論"). The paper says it was not tested.

## What changed in v3 (relative to v2, versions/v2_final_20261104/main.tex)
Four numbers were corrected after an independent recomputation of the paper's numbers from episodes (all double rounding, no conclusion changes):
- "12-14%" -> "11-14%" (H1a relative effect, s>=0.5). Original source: tests rel_effect range -0.115 to -0.144, read as 12-14%. True values: 11.48% (circle s=1) to 14.38%; 11.48 rounds to 11. Recomputed in recompute_v3_numbers.py block (1).
- "-0.17" -> "-0.16" (H4 circle slope without s=0, post hoc). Original source: scripts/recompute_h4_sens.py, circle formal "s>=0.5 only" row, -0.16493, which v2 wrote as -0.17. Correct rounding is -0.16.
- "31.9" -> "31.8" (formal DOB - CON-ATT at s=2.5, circle, in the H2 figure caption). Original source: tests DESC B_DOB s=2.5 mean_diff_cm = 31.8454 (PREREG_results.md prints +31.85); 31.85 rounded again gave 31.9. 27.2 was right.
- "2.4-3.7" -> "2.3-3.7" (CON-ATT tracking error range). Original source: cells CON-ATT speed 1.0, p_drop 0, min 2.3473 (circle s=0), max 3.7129; v2 wrote the minimum as 2.4.
Other changes:
- Packet loss p=0.9 now compares with the lossless run on the same 20 seeds (v2 used the 40-seed lossless run). v2 values: zeta +52/+46%, dm +34/+22%, tpe10 +33/+29%; v3 values: +41/+39%, +19/+14%, +23/+22% (Table IV; recompute_v3_numbers.py block (5)). The p=0.9 values themselves are unchanged, and the tracking-error comparison now reads 2.88/2.91 lossless vs 2.80/2.90.
- Failed episodes are described as a set (35 episodes, per-method counts, no non-finite values, last-second mean error 10.1-20.3 cm, maximum error <= 23 cm), recomputed in recompute_v3_numbers.py block (4). The count split (30/4/1) was previously taken from a progress log that is not in the public repo.
- Three post hoc analyses were added (ORACLE reference; message-layer completeness 99.7% / 39%; two-sided p, adjusted 0.060): sources are listed in "Post hoc analyses added in v3".
- Abstract and body no longer quote "-0.38 to -0.67"; the abstract gives the relative effect (11-14%) and the CON-ATT error range (2.3-3.7 cm). The -0.67 now appears only in the Table II caption and Appendix C.
- Figures renumbered: Fig. 1 scene, Fig. 2 ladder of methods (no data), Fig. 3 H1 figure (figs/fig_H1_v3.pdf), Fig. 4 H2 figure in Appendix C (figs/fig_H2_raw_v3.pdf); both drawn by scripts/make_figs_v3.py (v3 axis label and panel titles; formulas unchanged from make_figs.py, which is kept for v2).
- References to a private progress log were replaced by result files or scripts.

## Scripts
- paper/workshop/scripts/make_figs_v3.py: Fig. 3 and Fig. 4 (`.venv/bin/python paper/workshop/scripts/make_figs_v3.py`). make_figs.py is the v2 version.
- paper/workshop/scripts/recompute_v3_numbers.py: values of the v3 text that no other script produces (H1a relative effect, CON-ATT range, DOB - CON-ATT at s=2.5, failed-episode set, p=0.9 vs the same-seed lossless run). Prints each value with the quoted paper value and an OK/MISMATCH flag. Reads only results/prereg/episodes.jsonl through the frozen loader in experiments/prereg_analyze.py.
- paper/workshop/scripts/recompute_posthoc_extras.py: ORACLE reference, message-layer completeness, two-sided p-values.
- paper/workshop/scripts/recompute_medians_slopes.py: Table II medians and raw H2 slopes (run: `.venv/bin/python -I paper/workshop/scripts/recompute_medians_slopes.py`). Logic: key episodes by (method, profile, traj) for speed 1.0 and CON-ATT p_drop 0 (p_drop 0.5/0.9 share the method/profile/traj name; distinguish via hp["p_drop"]); RMSE_eff = 100 if not finite or not success or rmse>1 else 100*rmse.
- paper/workshop/scripts/recompute_h4_sens.py: H4 sensitivity slopes.
- paper/workshop/scripts/render_snapshot.py: Fig. 1.
