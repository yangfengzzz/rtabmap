# SuperPoint, SuperGlue, And D455 Workflow

This is the highest-value task path for future agent work in this repo: build RTAB-Map with SuperPoint and SuperGlue support, then validate that the build can be used with an Intel RealSense D455 through the RealSense2 backend.

## What Exists In The Repo Already

- SuperPoint Torch support exists behind `WITH_TORCH` in `CMakeLists.txt`.
- SuperGlue matcher mode `Vis/CorNNType=6` is a native libtorch backend under `corelib/src/superglue_torch/`.
- Upstream SuperGlue checkpoints should be converted once with `scripts/convert_superglue_weights.py`.
- The RealSense D400 family, including D455, should go through `CameraRealSense2`, not the legacy RealSense driver.
- `tools/Matcher/main.cpp` already shows a canonical SuperPoint + SuperGlue invocation example.
- `tools/CameraRGBD/main.cpp` exposes RealSense2 as driver `11`.

## Compile-Time Requirements

For the combined workflow, the minimum build toggles are:

```bash
cmake -S . -B build \
  -DWITH_TORCH=ON \
  -DWITH_REALSENSE2=ON
```

Interpretation:

- `WITH_TORCH=ON`
  - enables native SuperPoint Torch support
- `WITH_REALSENSE2=ON`
  - enables librealsense2-backed camera support for the D455

## Configure-Time Success Criteria

The CMake summary should indicate all of the following:

- `With SuperPoint = YES`
- `With SuperGlue = YES`
- `With Python3 = YES`
- `With RealSense2 = YES`

Nice to have, but not strictly required for SuperGlue:

- `With Superpoint Rpautrat = YES`

That Rpautrat path is a different SuperPoint integration and depends on both Torch and Python.

## Runtime Assets And External Dependencies

The repo contains wrappers, not all upstream model assets.

- SuperPoint Torch needs a `.pt` model file referenced by `SuperPoint/ModelPath`.
- SuperGlue needs a converted native weights file referenced by `SuperGlue/WeightsPath`.
- `scripts/convert_superglue_weights.py` converts upstream OrderedDict checkpoints into a plain-dict archive that libtorch C++ can read directly.

Implication for agents:

- If build succeeds but runtime matching fails, distinguish between:
  - missing compiled support
  - missing model file
  - missing Python module / wrong checkout layout

## Canonical Matching Validation

The fastest feature-stack validation is the matcher tool, because it avoids camera and mapping complexity.

The repo already documents this pattern in `tools/Matcher/main.cpp`:

```bash
rtabmap-matcher \
  --Vis/FeatureType 11 \
  --SuperPoint/ModelPath "superpoint.pt" \
  --Vis/CorNNType 6 \
  --SuperGlue/WeightsPath "superglue_indoor_native.pth" \
  from.png to.png
```

Interpretation:

- `Vis/FeatureType 11`
  - use SuperPoint Torch
- `Vis/CorNNType 6`
  - use native SuperGlue matcher mode
- `SuperGlue/WeightsPath`
  - point at the converted native checkpoint

In practice, also expect to set or verify:

- `SuperGlue/Cuda=true|false`
- `SuperGlue/Threshold`
- `SuperGlue/Iterations`

## D455 Validation Path

For camera bring-up, start with the narrow camera tool before trying the full app.

`tools/CameraRGBD/main.cpp` exposes:

- driver `11 = RealSense2`

Useful bring-up pattern:

```bash
rtabmap-cameraRGBD -d <serial-or-device> -w 640 -h 480 11
```

Notes:

- The tool uses the RealSense2 SDK path, which is the correct family for D455.
- If hardware enumeration fails, debug librealsense2 access first, not RTAB-Map mapping logic.
- The `CameraRealSense2` implementation contains configuration hooks for:
  - emitter enable/disable
  - IR mode
  - image and depth resolution
  - global time sync
  - dual mode / odometry-provided mode
  - JSON device config

These are the places to inspect when a D455 task mentions stream format, synchronization, or sensor configuration behavior.

## Relevant Source Anchors

- Build flags and summary: `CMakeLists.txt`
- Feature parameters: `corelib/include/rtabmap/core/Parameters.h`
- SuperPoint implementation: `corelib/src/Features2d.cpp`
- Native SuperGlue matcher: `corelib/src/superglue_torch/SuperGlue.cpp`
- Checkpoint converter: `scripts/convert_superglue_weights.py`
- RealSense2 backend: `corelib/src/camera/CameraRealSense2.cpp`
- Camera validation tool: `tools/CameraRGBD/main.cpp`
- Matcher validation tool: `tools/Matcher/main.cpp`
- GUI capability visibility: `guilib/src/AboutDialog.cpp`, `guilib/src/PreferencesDialog.cpp`

## Recommended Agent Debug Sequence

1. Verify `WITH_TORCH` and `WITH_REALSENSE2` in `build/CMakeCache.txt`.
2. Re-run CMake if any of them are disabled or stale.
3. Build `rtabmap-matcher` and `rtabmap-cameraRGBD` first.
4. Validate SuperPoint + SuperGlue offline with images.
5. Validate D455 camera bring-up separately.
6. Only after both pass, move to the full RTAB-Map app or mapping workflow.

## Frequent Failure Modes

- `WITH_TORCH=OFF`
  - SuperPoint selection falls back and warnings appear at runtime.
- librealsense2 not found
  - D455 support compiles out or reports unavailable at runtime.
- raw upstream checkpoint used directly
  - convert it first with `scripts/convert_superglue_weights.py` or the native loader will reject it.
- model assets missing
  - compiled support exists, but runtime fails when loading `.pt` or converted SuperGlue weights.

## What To Say In A Handoff

If you cannot access physical hardware, be explicit:

- build inclusion verified
- camera tool compiled
- RealSense2 backend present
- D455 runtime on actual hardware not exercised in this session

That distinction matters in this repo because build success and hardware success are separate milestones.
