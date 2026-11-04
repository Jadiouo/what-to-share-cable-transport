# results/media: video notes

Re-run (project root, CPU only, software GL via Mesa llvmpipe; the GPU is not used):

```
# Linux, headless; software GL via Mesa llvmpipe (no GPU needed)
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json LIBGL_ALWAYS_SOFTWARE=1 \
  .venv/bin/python experiments/render_videos.py all \
  [--only hero_con_s1 dob_failure_s25 geo_vs_local_s175 con_vs_geo_s0]
```
`select` prints the chosen episodes only; `record` re-runs and verifies the episodes (cache in `.scratch/render_videos/`); `render` makes the videos. The chosen episodes are also written to `results/media/selection.json`.

All videos are MuJoCo 3D renderings (the formal simulator; visual-only geoms are added to the same MJCF, see `paper/workshop/scripts/render_snapshot.py`) with a payload position-error plot underneath, |p - p_ref| in cm (grey = full curve, colour = played so far). Two-column videos use the same seed on both sides (same true parameters and initial state). Every frame shows the method name, the time, the RMSE of the whole episode (raw, in cm, not the 1 m-clipped score) and whether the episode succeeded. Red dots are the reference trajectory. Camera: az 60, el -16, dist 2.8, lookat between the payload and the reference-circle centre. Playback is real time (25 fps, one frame per 4 control steps of 0.01 s) with a 1 s hold at the end. MP4: H.264, yuv420p, 1080x720. GIF: 720 px wide, 12 fps, palettegen, each under 5 MB.

All videos: speed 1.0, seeds 3000-3039, frozen hp (`HP` in `experiments/prereg_run.py`), p_drop = 0 for CON-ATT, trajectory circle unless stated.

## Episode selection rule (fixed in advance, not based on how the video looks)

The sensing level of each comparison clip was chosen as the cell where the effect reaches the pre-registered meaningful size of −0.5 cm (H1a at s = 1.75, H1b at s = 0), or, for DOB, the harshest level s = 2.5. The rule below then picks the episode within that cell.

All numbers used for selection are read from `results/prereg/episodes.jsonl` (CON-ATT rows are filtered by the frozen hp, p_drop = 0).

- `hero_con_s1`: fixed, CON-ATT, s=1, circle, seed 3000 (single column overview).
- `dob_failure_s25`: among the DOB episodes with success = False at s=2.5 (circle), take the one whose raw RMSE is closest to the median RMSE of those failures (ties: smaller seed). If circle had no DOB failure, fig8 would be used; this was not needed.
- `geo_vs_local_s175` and `con_vs_geo_s0`: gap_i = rmse_left_i - rmse_right_i (raw) over the 40 seeds; the shown episode is argmin_i |gap_i - mean(gap)|, ties to the smaller seed.

Before a video is made, the selected episodes are re-run (a monkeypatch on `MultiPayloadEnv.step` records qpos; the simulation is unchanged) and the rerun RMSE must equal the formal one to 1e-9, and the success flag must agree, otherwise the script exits without producing any video.

## Videos

| File | Selected seed | Episode RMSE (left vs right, cm) | 40-episode mean (left vs right, cm) | gap (episode) | mean gap |
|---|---|---|---|---|---|
| `hero_con_s1` | 3000 | CON-ATT 2.577 | CON-ATT 2.798 | n/a | n/a |
| `dob_failure_s25` | 3013 | DOB 9.378 vs CON-ATT 3.234 | DOB 8.625 vs CON-ATT 3.675 | 6.14 | 4.95 |
| `geo_vs_local_s175` | 3013 | EKF-L 3.466 vs EKF-L+GEO 2.955 | EKF-L 3.902 vs EKF-L+GEO 3.399 | 0.511 | 0.503 |
| `con_vs_geo_s0` | 3004 | EKF-L+GEO 3.362 vs CON-ATT 2.709 | EKF-L+GEO 2.988 vs CON-ATT 2.347 | 0.653 | 0.641 |

(gap = left - right; the mean in the "mean" columns is over all 40 seeds, including failed ones.)

### `hero_con_s1.mp4/.gif`
- Content: CON-ATT, s=1, circle, single column, as an overview. Seed fixed to 3000 in advance; no gap-based selection. 40-episode mean for context only.

### `dob_failure_s25.mp4/.gif`
- Content: left DOB, right CON-ATT, s=2.5, circle. DOB fails (success = False) in 12 of 40 seeds on circle; the median RMSE of those 12 failures is 9.62 cm (even count, so it is the mean of the two middle values) and seed 3013 (9.38 cm) is the closest failure (failed seeds: 3003, 3005, 3006, 3007, 3013, 3014, 3016, 3019, 3020, 3022, 3029, 3031).
- Note that success requires a last-second mean error below 10 cm and a maximum error below 50 cm; here the DOB error is mostly 6-13 cm (minimum about 3.5 cm near t = 1.75 s) and never diverges, so the picture shows a persistent offset, not a crash.

### `geo_vs_local_s175.mp4/.gif`
- Content: left EKF-L, right EKF-L+GEO, s=1.75, circle. Seed 3013 has gap +5.11 mm against a mean gap of +5.03 mm.
- The difference is millimetre-scale: both quadrotor teams track the circle and look almost identical in the 3D view. This is how the effect actually looks; it is not a rendering error. Only the error plot shows the difference.

### `con_vs_geo_s0.mp4/.gif`
- Content: left EKF-L+GEO, right CON-ATT, s=0, circle. Seed 3004 has gap +6.53 mm against a mean gap of +6.41 mm.
- Same remark: a sub-centimetre difference, the two 3D scenes look nearly the same, shown as it is.

## Verification

Rerun RMSE vs `results/prereg/episodes.jsonl` (all 7 episodes shown in the videos; tolerance 1e-9; success flags also equal):

| Method | Profile | Seed | Rerun RMSE | Formal RMSE | abs diff |
|---|---|---|---|---|---|
| CON-ATT | s=1 | 3000 | 0.025769449297 | 0.025769449297 | 0 |
| DOB | s=2.5 | 3013 | 0.093778481734 | 0.093778481734 | 0 |
| CON-ATT | s=2.5 | 3013 | 0.032337289497 | 0.032337289497 | 0 |
| EKF-L | s=1.75 | 3013 | 0.034662646686 | 0.034662646686 | 0 |
| EKF-L+GEO | s=1.75 | 3013 | 0.029547776561 | 0.029547776561 | 0 |
| EKF-L+GEO | s=0 | 3004 | 0.033619033615 | 0.033619033615 | 0 |
| CON-ATT | s=0 | 3004 | 0.027091448758 | 0.027091448758 | 0 |

The error curve drawn in the videos is recomputed from the recorded payload position and the reference (same definition as the episode runner); its RMSE equals the formal RMSE to better than 1e-16.
