# 3DGS Export Notes

This repo now has a dedicated 3DGS dataset exporter:

```bash
python3 scripts/export_3dgs_dataset.py --db /path/to/map.db
```

It writes:

- `images/` with exported RGB frames
- `sparse/0/cameras.txt`
- `sparse/0/images.txt`
- `sparse/0/points3D.txt`
- `transforms.json`
- `points3D.ply`

## Implementation Notes

- The user-facing entry point is [`scripts/export_3dgs_dataset.py`](../../scripts/export_3dgs_dataset.py).
- The heavy lifting is done by the native helper [`tools/Export3DGS/main.cpp`](../../tools/Export3DGS/main.cpp), built as `build/bin/rtabmap-export3dgs_dataset`.
- Sparse points/tracks come from `Optimizer::computeBACorrespondences()`, mirroring RTAB-Map's Bundler export logic.

## Calibration Gotcha

Not every RTAB-Map database stores full raw distortion metadata.

Two valid cases exist:

- Full calibration:
  - `CameraModel::isValidForRectification() == true`
  - exporter rectifies the RGB image and rectifies 2D observations
- Minimal pinhole calibration only:
  - `CameraModel::isValidForProjection() == true`
  - `CameraModel::isValidForRectification() == false`
  - exporter treats the image as already usable pinhole input and exports it as-is

That fallback is intentional. It avoids rejecting databases that only store `fx/fy/cx/cy/size/localTransform`, which is common in some RTAB-Map capture paths.

## Scope

- V1 is single RGB-D camera only.
- V1 is designed for D455-style RTAB-Map databases.
- Stereo-only and multi-camera sessions are intentionally rejected.
