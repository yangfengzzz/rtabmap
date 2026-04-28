# SuperPoint, SuperGlue, And D455 Workflow

This is the highest-value task path for future agent work in this repo: build RTAB-Map with SuperPoint and SuperGlue support, then validate that the build can be used with an Intel RealSense D455 through the RealSense2 backend.

## What Exists In The Repo Already

- SuperPoint Torch support exists behind `WITH_TORCH` in `CMakeLists.txt`.
- SuperGlue is integrated as a Python matcher bridge, not a native C++ matcher.
- The bundled wrapper lives at `corelib/src/python/rtabmap_superglue.py`.
- The RealSense D400 family, including D455, should go through `CameraRealSense2`, not the legacy RealSense driver.
- `tools/Matcher/main.cpp` already shows a canonical SuperPoint + SuperGlue invocation example.
- `tools/CameraRGBD/main.cpp` exposes RealSense2 as driver `11`.

## Compile-Time Requirements

For the combined workflow, the minimum build toggles are:

```bash
cmake -S . -B build \
  -DWITH_TORCH=ON \
  -DWITH_PYTHON=ON \
  -DWITH_REALSENSE2=ON
```

Interpretation:

- `WITH_TORCH=ON`
  - enables native SuperPoint Torch support
- `WITH_PYTHON=ON`
  - enables `PyMatcher` and embedded Python integration used by SuperGlue
- `WITH_REALSENSE2=ON`
  - enables librealsense2-backed camera support for the D455

## Configure-Time Success Criteria

The CMake summary should indicate all of the following:

- `With SuperPoint = YES`
- `With Python3 = YES`
- `With RealSense2 = YES`

Nice to have, but not strictly required for SuperGlue:

- `With Superpoint Rpautrat = YES`

That Rpautrat path is a different SuperPoint integration and depends on both Torch and Python.

## Runtime Assets And External Dependencies

The repo contains wrappers, not all upstream model assets.

- SuperPoint Torch needs a `.pt` model file referenced by `SuperPoint/ModelPath`.
- `corelib/src/python/rtabmap_superglue.py` is intended to be copied into a SuperGlue checkout.
- That wrapper imports `models.matching.SuperGlue`, so the runtime Python environment must be able to resolve the upstream SuperGlue package layout.

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
  --PyMatcher/Path "~/SuperGluePretrainedNetwork/rtabmap_superglue.py" \
  from.png to.png
```

Interpretation:

- `Vis/FeatureType 11`
  - use SuperPoint Torch
- `Vis/CorNNType 6`
  - use Python matcher mode
- `PyMatcher/Path`
  - point at the wrapper inside the external SuperGlue checkout

In practice, also expect to set or verify:

- `PyMatcher/Model=indoor` or `outdoor`
- `PyMatcher/Cuda=true|false`
- `PyMatcher/Threshold`
- `PyMatcher/Iterations`

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
- SuperGlue wrapper: `corelib/src/python/rtabmap_superglue.py`
- Python bridge: `corelib/include/rtabmap/core/PythonInterface.h`
- RealSense2 backend: `corelib/src/camera/CameraRealSense2.cpp`
- Camera validation tool: `tools/CameraRGBD/main.cpp`
- Matcher validation tool: `tools/Matcher/main.cpp`
- GUI capability visibility: `guilib/src/AboutDialog.cpp`, `guilib/src/PreferencesDialog.cpp`

## Recommended Agent Debug Sequence

1. Verify `WITH_TORCH`, `WITH_PYTHON`, and `WITH_REALSENSE2` in `build/CMakeCache.txt`.
2. Re-run CMake if any of them are disabled or stale.
3. Build `rtabmap-matcher` and `rtabmap-cameraRGBD` first.
4. Validate SuperPoint + SuperGlue offline with images.
5. Validate D455 camera bring-up separately.
6. Only after both pass, move to the full RTAB-Map app or mapping workflow.

## Frequent Failure Modes

- `WITH_TORCH=OFF`
  - SuperPoint selection falls back and warnings appear at runtime.
- `WITH_PYTHON=OFF`
  - `PyMatcher` path is unavailable even if SuperPoint is built.
- librealsense2 not found
  - D455 support compiles out or reports unavailable at runtime.
- wrapper copied incorrectly
  - `PyMatcher/Path` points to a file that cannot import `models.matching.SuperGlue`.
- model assets missing
  - compiled support exists, but runtime fails when loading `.pt` or Python-side weights.

## What To Say In A Handoff

If you cannot access physical hardware, be explicit:

- build inclusion verified
- camera tool compiled
- RealSense2 backend present
- D455 runtime on actual hardware not exercised in this session

That distinction matters in this repo because build success and hardware success are separate milestones.
