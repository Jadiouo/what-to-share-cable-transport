# NUMBERS: where each number in main.tex comes from
Paths relative to repo root. "tests" = results/prereg/tests.csv (family F1 unless noted; col mean_diff_cm, ci95_*, p_holm_adj, rel_effect). "sens" = results/prereg/sensitivity.csv. "cells" = results/prereg/cells.csv. "RECOMPUTED" = not in any result file; computed from results/prereg/episodes.jsonl by paper/workshop/scripts/ (see bottom); verify before submission.

## Design / protocol (text, not results)
- commit 5dd3cf7, seeds 3000-3039, no deviations, 3240 episodes: docs/PREREG.md header (frozen version) and results/prereg/PREREG_results.md.
- N=3, 1 kg quad, payload 1 kg, r=0.15 m, rope 1 m, dt 0.01: src/smpc/envs/params.py, src/smpc/eval/episode.py (M_NOM, PAYLOAD_RADIUS), multi_payload.py default rope_length=1.0.
- dm in [-0.3,0.4], |dc|<=0.04: src/smpc/eval/episode_est.py DM_RANGE, DC_MAX.
- MPPI K=512, H=20, dt_plan 0.05: configs/mppi_frozen.json.
- circle R=0.8, w=1.2; fig8 A=0.8,B=0.4,w=0.9; 10 s: src/smpc/control/trajectories.py, eval/episode.py DURATION.
- s scaling (sigma_a 0.15+0.45s, bias 0.1s, kerr 0.05s, delay 0/2/3 at s<0.5/<1.75/else, pos_std 0.01 min(s,1)): src/smpc/eval/sensing.py (PROFILES, scaled_profile).
- Method x information table, method descriptions: src/smpc/adapt/ekf_geo.py header; docs/PREREG.md 3.1; ekf_local.py; con_att.py (HCMCI, iters=2, path graph, relay N-1, max_age=5, dropout rule). use_dc=0 for EKF-L+GEO: docs/PREREG.md 3.2.
- RMSE_eff definition, success definition, 1 m rule: docs/PREREG.md section 4.
- sign-flip 1e5, bootstrap 1e4, Holm families F1-F8, TOST +-0.25 (+-0.5 sensitivity), Delta=0.5 cm: docs/PREREG.md sections 1, 5.
- Tuning seeds 2020-2024, s=1, speed 1.0; budget 14/10/8; boundary optima (LOC-M lam, CEN-ATT prior): docs/PREREG.md 3.3.
- Pilot power ~0.1 for H1b s>=1.75: docs/PREREG.md section 5 ("檢定力").

## Fig. 1 (simulator scene)
- figs/sim_snapshot_t4.png: paper/workshop/scripts/render_snapshot.py (record: reruns CON-ATT, s=1, circle, seed 3000 with frozen hp; RMSE 0.025769 identical to episodes.jsonl; render: CPU llvmpipe, frame t=4 s, az 60, el -16, dist 2.8).

## Abstract / Table II / H1 text
- Table II means: tests F1 mean_diff_cm (rounded 2 dp), all 20 rows. Italic = rejected False in tests (H1a s=0 both; H1b s=1.75, 2.5 both).
- Table II medians: RECOMPUTED (paired differences of RMSE_eff per seed, np.median).
- "-2.97" fig8 s=2.5 Da: tests (H1a|s=2.5|fig8). "-0.67 (90% CI -0.96,-0.38)": sens a_p2d_rm_rule, H1a|s=2.5|fig8 (mean -0.665202, ci90 -0.96229,-0.377384). Total -3.20 and H4 fig8 -1.07: PREREG_results.md (TOTAL row; H4 row) / tests F5. "Other 19 cells unchanged": verified by diffing sens(a) mean_diff_cm against tests (only 1 differs).
- "-0.61, n=39" dropping failed pairs: sens c_drop_failed_pairs H1a|s=2.5|fig8 (-0.605963, n=39).
- "12-14% of EKF-L RMSE": tests rel_effect for H1a s>=0.5, excluding fig8 s=2.5 (range -0.115 to -0.144).
- "Holm-adjusted p 0.001 to 0.033" for H1a s>=0.5: tests p_holm_adj (min 0.0011, max 0.0329).
- "H1b adjusted p 0.07-0.21" at s=1.75, 2.5: tests (0.1872, 0.0707, 0.2050, 0.1571).
- "-0.16 to -0.64" (H1b s<=1) : tests (-0.16..-0.64). "only s=0 reaches Delta; medians -0.27, -0.14": tests delta_layer; medians RECOMPUTED.
- "Both trajectories <= -0.5 only at s=1.75, 2.5": PREREG_results.md Delta column / tests delta_layer. "Medians -0.20..-0.45": RECOMPUTED.
- H4 slopes -0.40, -1.07, p 0.0004, 0.0029: tests F5 (mean_diff_cm, p_holm_adj 0.00038, 0.0029).
- Seed 3024, EKF-L s=2.5 fig8, rmse 0.0792 m, success False: episodes.jsonl key prereg|EKF-L|s=2.5|fig8|1.0|3024|q_b=0.0005,r_scale=10.
- Raw RMSE EKF-L ~ cells.csv column rmse_eff_mean_cm (not needed in text).

## H2
- +11.0, +10.6, p<1e-4: tests F2 (11.0298, 10.5674; p_holm_adj 1.99998e-05). "all 40 seeds slope>0": h2_h4_slopes.csv (verified both RMSE_eff and raw: 40/40, RECOMPUTED for raw).
- 22/80 DOB failures at s=2.5, 58/80: cells success_rate (0.70, 0.75 x 40) ; Wilson 0.70 [0.546,0.819], 0.75 [0.598,0.858]: cells succ_ci_lo/hi. CON-ATT 80/80: cells.
- Raw slopes +1.84 (circle), +1.90 (fig8): RECOMPUTED. Raw B_DOB means 0.40/0.22 at s=0 and 4.95/4.92 at s=2.5: RECOMPUTED. Fig. 2 values: same, bootstrap CI generated in scripts/make_figs.py (not preregistered).
- 31.9 and 27.2 cm formal B_DOB at s=2.5: PREREG_results.md descriptive B_DOB table (+31.85, +27.23).
- LOC-M slope +1.34, +2.89: tests F4. LOC-M fails 4 at s=2.5: cells (0.975 and 0.925 x 40 = 1 + 3).

## H3
- -0.021, -0.033 at s=1; p<1e-4: tests F3 (p_holm_adj 1.99998e-05). Means -0.049..-0.005 over five s: tests F3+F6 (min -0.0491 fig8 s=2.5; max -0.0046 circle s=0). +-0.5 band passes: sens b_tost_0.5cm (all rejected True).
- "Nominally slightly better" (CON-ATT mean < CEN-ATT mean): sign of D3 above, cells rmse_eff_mean_cm.

## Packet loss
- -0.58, -0.64, p 1.4e-4, 8e-5: tests F8 (-0.577413, -0.640494; p_holm_adj 0.00013999, 7.99992e-05).
- RMSE 2.76/2.83 (p=0.5) vs 2.80/2.85 (p=0): cells (CON-ATT s=1 p_drop 0.5 and 0). p=0.9, 20 seeds: cells (2.80/2.90); overconfidence pilot remark: docs/PREREG.md section 2.
## Speed 1.5
- H1a@1.5 circle +0.18 (n.s.), fig8 -0.22; H1b@1.5 -0.37, -0.18: tests F7.

## Limitations
- RMSE range 2.4-3.7 cm: cells, CON-ATT speed 1.0 min 2.35, max 3.71.
- Holm p 0.02-0.03: tests p_holm_adj (H1a s=1: .0328,.0217; H1b s=1: .0303; H1a s=1.75: .0199).
- Pilot Da(s=1) -0.52/-0.50 vs formal -0.38/-0.42: results/prereg/power.md (-0.520, -0.496) and tests.
- 35 failures total (DOB 30, LOC-M 4, EKF-L 1): counted from episodes.jsonl (success False = 35; sens note "=35"); split by method from the same file.

## Scripts
- paper/workshop/scripts/make_figs.py (figures). RECOMPUTED medians and raw slopes were computed with scripts/recompute_medians_slopes.py (run: .venv/bin/python -I paper/workshop/scripts/recompute_medians_slopes.py); logic: key episodes by (method, profile, traj) for speed 1.0 and CON-ATT p_drop 0 (CON-ATT p_drop=0.5/0.9 cells share the same method/profile/traj name; distinguish via hp["p_drop"]); RMSE_eff = 100 if not finite or not success or rmse>1 else 100*rmse.

## Revision 2 additions
- Tuning budget 14/10/10/9/8 and "10 episodes per hp point (seeds 2020-2024 x 2 traj)": docs/PREREG.md 3.3 table (lines ~113-125). EKF-L extension = deviation from p2d_prespec, disclosed there.
- Freeze = commit 5dd3cf7, doc v3 (PREREG header); pilot seeds 2200-2219 (also 2020-2029, 2100-2119): PREREG sec 2 last bullets. "Failed = 1 m rule set after pilot": PREREG sec 4, bullet "與 pilot 的差異" (line ~137).
- use_dc=0 for GEO: results/p2d/best.json (EKF-L+GEO hp); ekf_geo.py DEFAULTS has use_dc=1 but frozen hp overrides it. CON-ATT estimates dc (state [dm, dc, zeta_1..N]).
- H3 90% CI: tests.csv ci90 cols, F3 rows: circle [-0.0320, -0.0098], fig8 [-0.0431, -0.0227].
- "+-0.25 band same order as H1b (0.16-0.64)": comparison of tests F3 band vs F1 H1b means.
- Relay N-1 rounds / drop also applies to relay rounds / stale -> nominal geometry: src/smpc/adapt/con_att.py (docstring lines ~25-30, p_drop applied each iteration incl. relay).
- How MPPI uses estimates (theta_hat_i into team.set_theta, eta=1+zeta_hat): src/smpc/eval/episode_p2c.py lines ~138-172.
- s=1.75 H1a means -0.503 (circle), -0.513 (fig8): tests F1 (and sens a).
- Abstract "-0.38 to -0.67": sens a_p2d_rm_rule H1a s>=0.5 (min -0.378, max -0.665 fig8 s=2.5).
- H4 sensitivity: RECOMPUTED by scripts/recompute_h4_sens.py. Formal -0.401 (circle), -1.069 (fig8). Sensitivity (a): circle -0.401 (unchanged), fig8 -0.283. Excluding s=0 end point: circle -0.165; fig8 formal -1.161, (a) -0.096. (sensitivity.csv has no H4 rows.)
- Packet loss p=0.9 (revision 3, replaces the earlier REMOVED note): cells.csv columns zeta_rms_h2_mean, dm_err5_mean, tpe10_mean, rows CON-ATT,s=1,speed 1, p_drop 0 (n=40) vs 0.9 (n=20), circle/fig8. zeta 0.0128317/0.0133228 -> 0.0194803/0.0194172; dm 0.0394512/0.0443772 -> 0.0529164/0.0541889; tpe10 0.218848/0.22632 -> 0.292087/0.292469. Percent changes RECOMPUTED (b/a-1): zeta +51.8/+45.7; dm +34.1/+22.1; tpe10 +33.5/+29.2 (text rounds to +52/+46, +34/+22, +33/+29). RMSE p=0.9: cells 2.80336/2.90421 (text 2.80/2.90); p=0: 2.79814/2.85178.
- CON-ATT p=0.9 vs EKF-L+GEO (20 seeds, descriptive, no Holm): tests.csv DESC rows DROP0.9-vs-GEO|s=1: circle -0.250669 (95% CI -0.478698,-0.0481378), fig8 -0.213278 (-0.406646,-0.0417068).
- Revision 3: H1a range unified to -0.38..-0.67 in abstract (-0.67 = fig8 s=2.5 under sens (a)); body quotes -0.38..-0.65 for non-artifact cells and notes the -0.67. H4 fig8: formal -1.069 (p 0.0029), sens (a) -0.283, (a) minus s=0 -0.096: scripts/recompute_h4_sens.py (RECOMPUTED).
- HYB and ORACLE in the 3240 episodes: descriptive only (PREREG G5).
- Related-work citations in refs.bib are complete entries, checked against Crossref/arXiv on 2026-11-01.
