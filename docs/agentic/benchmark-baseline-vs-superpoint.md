# Baseline Vs Existing SuperPoint + LightGlue Preset

Use this workflow when comparing default RTAB-Map visual frontend behavior against the existing SuperPoint + LightGlue preset on the same D455 setup.

## Launchers

- Baseline:
  `./scripts/run_rtabmap_d455_default_baseline.sh`
- SuperPoint + LightGlue:
  `./scripts/run_rtabmap_d455_superpoint_lightglue.sh`

These two launchers keep the same D455 / RealSense2 source path. The baseline launcher leaves RTAB-Map visual settings at defaults. The existing SuperPoint + LightGlue launcher uses the tuned preset already in this repo.

## Fair Test Method

1. Start with the baseline launcher and scan the same route you plan to use for the SuperPoint + LightGlue run.
2. Save the database with a clear name, for example:
   - `baseline_YYYYmmdd-HHMMSS.db`
3. Start the existing SuperPoint + LightGlue launcher and scan the same route again at similar speed.
4. Save the database with a matching comparison name, for example:
   - `superpoint_lightglue_YYYYmmdd-HHMMSS.db`
5. Compare them with:

```bash
python3 scripts/compare_rtabmap_runs.py \
  ~/Documents/RTAB-Map/baseline_YYYYmmdd-HHMMSS.db \
  ~/Documents/RTAB-Map/superpoint_lightglue_YYYYmmdd-HHMMSS.db
```

## Metrics Reported

- `Nodes`
- `Statistics rows`
- `Duration (s)`
- `Neighbor links`
- `Loop links`
- `Avg words/frame`
- `Avg visual matches`
- `Avg visual inliers`
- `Avg inlier ratio`
- `Avg total time (ms)`
- `Max total time (ms)`
- `Avg memory update`
- `Avg hyp creation`
- `Avg hyp validation`
- `Avg keypoint detect`
- `Avg descriptor time`

## Interpretation

- Lower timing numbers mean faster frontend/backend processing.
- Higher `Avg visual inliers` and `Avg inlier ratio` usually indicate more stable visual registration.
- Higher `Nodes` can simply mean more frequent keyframe creation, not necessarily a better map.
- `Loop links` are useful to compare, but a larger number is not automatically better if timing or stability degrades.
- Always compare runs on the same route. Human motion differences can dominate small frontend differences.
- Because the existing SuperPoint + LightGlue preset is also tuned for steadier runtime behavior, the result reflects both the learned frontend and those tuning changes together.
