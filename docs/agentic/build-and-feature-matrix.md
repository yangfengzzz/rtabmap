# Build And Feature Matrix For Agents

This guide is for answering two questions quickly:

1. Which compile-time options control the backend I care about?
2. How do I verify that the configured build actually contains it?

## Core Build Facts

- Build system: CMake (`CMakeLists.txt`)
- Language center: C++ with optional Python embedding
- Important default options:
  - `BUILD_APP=ON`
  - `BUILD_TOOLS=ON`
  - `BUILD_EXAMPLES=ON`
  - `WITH_REALSENSE2=ON`
  - `WITH_TORCH=OFF`
  - `WITH_PYTHON=OFF`

The last two defaults matter: a stock configure will not include the SuperPoint/SuperGlue path unless explicitly enabled.

## Feature Matrix

| Capability | Compile-Time Options | Main Dependency | Main Source Anchors | Typical Runtime Surface |
| --- | --- | --- | --- | --- |
| SuperPoint Torch | `WITH_TORCH=ON` | libtorch | `CMakeLists.txt`, `corelib/src/Features2d.cpp` | `Vis/FeatureType 11`, `SuperPoint/*` |
| SuperPoint Rpautrat | `WITH_TORCH=ON`, `WITH_PYTHON=ON` | libtorch + Python3 | `corelib/src/Features2d.cpp`, `corelib/src/python/rtabmap_superpoint_rpautrat.py` | `Vis/FeatureType 16`, `SuperPointRpautrat/*` |
| SuperGlue via PyMatcher | `WITH_PYTHON=ON` | Python3 + external SuperGlue checkout | `corelib/src/python/rtabmap_superglue.py`, `RegistrationVis.*` | `Vis/CorNNType 6`, `PyMatcher/*` |
| RealSense2 / D455 | `WITH_REALSENSE2=ON` | librealsense2 | `corelib/src/camera/CameraRealSense2.cpp`, `cmake_modules/FindRealSense2.cmake` | `CameraRealSense2`, tool driver `11` |
| GUI capability display | usually `WITH_QT=ON` | Qt | `guilib/src/AboutDialog.cpp`, `guilib/src/PreferencesDialog.cpp` | About and Preferences dialogs |

## Configure-Time Checks

During CMake configure, inspect the emitted feature summary lines. For this repo, the most important ones are:

- `With SuperPoint`
- `With Superpoint Rpautrat`
- `With Python3`
- `With RealSense2`

If a task depends on one of those and the configure summary says `NO`, do not debug runtime behavior yet. Fix configuration first.

## Fast Verification Commands

Use these patterns after configuring a build tree.

Check cached options:

```bash
rg "WITH_(TORCH|PYTHON|REALSENSE2)" build/CMakeCache.txt
```

Check whether CMake found the right SDKs:

```bash
rg "Torch|Python3|realsense2|RealSense2" build/CMakeCache.txt
```

Re-run configure with explicit options when needed:

```bash
cmake -S . -B build \
  -DWITH_TORCH=ON \
  -DWITH_PYTHON=ON \
  -DWITH_REALSENSE2=ON
```

Compile a narrow target set first when possible:

```bash
cmake --build build --target rtabmap-matcher rtabmap-cameraRGBD
```

If target names differ in a given generator, inspect generated targets rather than guessing.

## Runtime Parameter Map

The parameters you usually need are defined in `corelib/include/rtabmap/core/Parameters.h`.

- SuperPoint Torch:
  - `Vis/FeatureType 11`
  - `SuperPoint/ModelPath`
  - `SuperPoint/Threshold`
  - `SuperPoint/NMS`
  - `SuperPoint/NMSRadius`
  - `SuperPoint/Cuda`

- SuperPoint Rpautrat:
  - `Vis/FeatureType 16`
  - `SuperPointRpautrat/WeightsPath`
  - `SuperPointRpautrat/ModelPath`

- SuperGlue / PyMatcher:
  - `Vis/CorNNType 6`
  - `PyMatcher/Path`
  - `PyMatcher/Iterations`
  - `PyMatcher/Threshold`
  - `PyMatcher/Cuda`
  - `PyMatcher/Model`

## Capability Exposure Locations

When a user says "the option isn't available", check where the capability should surface:

- Build summary: `CMakeLists.txt`
- Core runtime capability text: `corelib/src/Parameters.cpp`
- GUI About dialog labels: `guilib/src/AboutDialog.cpp`
- GUI Preferences widgets: `guilib/src/PreferencesDialog.cpp`

This separation helps distinguish:

- configure failure
- compiled-out backend
- runtime missing asset/model/script
- UI not exposing an already-built capability

## Reuse vs Fresh Build Trees

Prefer a fresh configure when:

- switching `WITH_TORCH` or `WITH_PYTHON` from `OFF` to `ON`
- switching between dependency installations
- debugging a suspected stale cache issue

Reusing the existing `build/` tree is fine when:

- the needed options are already enabled
- the dependency paths are unchanged
- the task is a source-only change in a known-good configuration

## Practical Debug Order

1. Verify compile-time option values.
2. Verify SDK discovery in the cache and configure output.
3. Build a narrow validation target.
4. Run the narrow validation path.
5. Only then debug the full app or GUI.
