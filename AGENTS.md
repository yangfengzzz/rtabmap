# AGENTS.md

This repository is large, dependency-heavy, and easy to misread if you start changing code before locating the right backend. Use this file as the default operating guide for autonomous agents working in `/home/yangfengzzz/Desktop/rtabmap`.

## Mission

- Make small, verifiable changes.
- Prefer reading existing build/runtime paths over inventing new ones.
- Preserve cross-platform behavior unless the task explicitly scopes the change to one platform.

## Start Here

1. Read [README.md](README.md) for project positioning.
2. Read [docs/agentic/repo-map.md](docs/agentic/repo-map.md) for subsystem boundaries.
3. Read [docs/agentic/build-and-feature-matrix.md](docs/agentic/build-and-feature-matrix.md) before touching CMake or optional backends.
4. If the task mentions SuperPoint, SuperGlue, Python matchers, RealSense2, or D455, also read [docs/agentic/superpoint-superglue-d455.md](docs/agentic/superpoint-superglue-d455.md).

## Repo Shape

- `corelib/`: core SLAM, features, registration, camera drivers, parameters.
- `guilib/`: Qt GUI, preferences, about dialog, runtime capability display.
- `app/`: main desktop/mobile app packaging and launch surfaces.
- `tools/`: many focused CLIs used for validation and debugging.
- `examples/`: smaller camera/mapping entry points.
- `cmake_modules/`: custom dependency discovery, including RealSense and other sensor SDKs.
- `.devcontainer/` and `docker/`: reproducible environments; inspect before assuming local machine state.
- `build/`: may already contain a partially configured tree. Reuse carefully instead of assuming it matches the current task.

## High-Signal File Anchors

- Top-level options and dependency detection: `CMakeLists.txt`
- Feature type and parameter definitions: `corelib/include/rtabmap/core/Parameters.h`
- Feature backend selection and SuperPoint implementations: `corelib/src/Features2d.cpp`
- Python matcher/detector interface: `corelib/src/python/`, `corelib/include/rtabmap/core/PythonInterface.h`
- RealSense2 camera integration: `corelib/include/rtabmap/core/camera/CameraRealSense2.h`, `corelib/src/camera/CameraRealSense2.cpp`
- GUI exposure of optional capabilities: `guilib/src/PreferencesDialog.cpp`, `guilib/src/AboutDialog.cpp`
- Fast CLI validation path for matching: `tools/Matcher/main.cpp`
- Fast CLI validation path for RGB-D cameras: `tools/CameraRGBD/main.cpp`

## Operating Rules For Agents

- Search before editing. Start with `rg` on the feature, camera, or parameter name.
- When a task concerns an optional backend, trace it through three layers before editing:
  1. CMake option and dependency detection
  2. Core implementation or abstraction
  3. User-facing exposure in GUI, tools, or parameters
- Treat `WITH_TORCH` and `WITH_REALSENSE2` as the core concerns for the native SuperPoint/SuperGlue + D455 path. `WITH_PYTHON` may still matter for other optional Python features, but it is no longer required for `Vis/CorNNType=6`.
- Do not assume the existing `build/` directory has the required options enabled. Verify with `CMakeCache.txt` or rerun configuration.
- Do not remove or simplify platform guards (`#ifdef RTABMAP_*`, CMake options, SDK checks) unless the task explicitly requires it.
- Prefer validating with a focused tool or example before attempting full app runtime verification.

## Build And Verification Heuristics

- CMake configure first, compile second. Optional backend failures are often obvious at configure time.
- For feature/backend work, inspect CMake status lines for:
  - `With SuperPoint`
  - `With Superpoint Rpautrat`
  - `With Python3`
  - `With RealSense2`
- For runtime checks, favor:
  - `rtabmap-matcher` for SuperPoint + SuperGlue matching
  - `rtabmap-cameraRGBD` for D455 camera bring-up
  - GUI About/Preferences dialogs only after CLI-level verification
- If hardware is unavailable, still verify build inclusion and document the remaining runtime gap explicitly.

## Common Pitfalls

- SuperGlue matcher mode `Vis/CorNNType=6` is now a native libtorch backend configured through `SuperGlue/*`, not `PyMatcher/*`.
- Upstream SuperGlue checkpoints should be converted once with `scripts/convert_superglue_weights.py` before using them with the native backend.
- `WITH_PYTHON=ON` may still be useful for other Python features and Rpautrat-based integration, but it is not required for native SuperGlue.
- D455 uses the RealSense2 path, not the legacy RealSense driver.
- Parameter names may have legacy aliases. Check `corelib/src/Parameters.cpp` before assuming a parameter is unused or dead.
- GUI availability labels are compile-time/runtime reflections, so a missing capability may be a build problem rather than a UI problem.

## Preferred Agent Workflow

1. Map the task to the right subsystem.
2. Identify the compile-time option(s) and runtime parameter(s).
3. Make the smallest change that preserves current architecture.
4. Reconfigure/build only the necessary target(s) if possible.
5. Verify with the narrowest relevant CLI or test path.
6. Record any unverified hardware or environment assumptions in the final handoff.

## Documentation Policy

- Keep `AGENTS.md` short and operational.
- Put detailed backend or workflow notes under `docs/agentic/`.
- When learning a repo-specific build/runtime trick that an agent would likely rediscover later, add it to docs instead of leaving it only in chat history.
