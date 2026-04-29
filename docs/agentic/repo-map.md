# RTAB-Map Repo Map For Agents

This is a working map of the repository for autonomous agents. It focuses on where behavior actually lives, not on exhaustive source coverage.

## Top-Level Layout

- `CMakeLists.txt`
  - Global build options, dependency discovery, install/export behavior, feature summaries.
- `package.xml`
  - ROS/cmake package metadata, useful for packaging context but not the main source of build truth.
- `corelib/`
  - Core SLAM library and most runtime behavior.
- `guilib/`
  - Qt GUI layer and capability exposure.
- `app/`
  - Standalone app build and platform packaging.
- `tools/`
  - Narrow CLIs that are often the fastest validation path.
- `examples/`
  - Smaller runnable programs for camera/mapping scenarios.
- `cmake_modules/`
  - Custom `find_package()` logic for SDKs not reliably handled by system packages.
- `docker/`, `.devcontainer/`
  - Reproducible environments and dependency recipes.
- `archive/`
  - Historical experiments and papers. Useful for examples, but not usually the first place to edit current behavior.

## `corelib/`

This is the center of gravity for most feature work.

- `corelib/include/rtabmap/core/Parameters.h`
  - Canonical parameter keys, defaults, help text, and enumerated feature/camera modes.
- `corelib/src/Parameters.cpp`
  - Parameter group behavior, capability summaries, and legacy parameter migration.
- `corelib/src/Features2d.cpp`
  - Feature detector/descriptor selection, including SuperPoint Torch and SuperPoint Rpautrat.
- `corelib/src/python/`
  - Python bridge scripts such as `rtabmap_superglue.py` and related wrappers.
- `corelib/include/rtabmap/core/PythonInterface.h`
  - C++ side of embedded Python integration.
- `corelib/src/camera/`
  - Sensor SDK integrations, including `CameraRealSense2.cpp`.
- `corelib/src/Rtabmap.cpp`, `RegistrationVis.*`, `Odometry*`
  - Important when behavior changes move beyond feature extraction and into registration or odometry.

## `guilib/`

Use this when a task mentions the desktop UI, parameter widgets, or capability visibility.

- `guilib/src/PreferencesDialog.cpp` and `guilib/src/ui/preferencesDialog.ui`
  - User-editable configuration surfaces for sensors, SuperPoint, native SuperGlue, and related options.
- `guilib/src/AboutDialog.cpp` and `guilib/src/ui/aboutDialog.ui`
  - Runtime display of compiled capability flags like SuperPoint, Python, and RealSense2.
- `guilib/src/MainWindow.*`
  - Application-level entry points and camera/source selection actions.

## `app/`

Use this when tasks involve the main shipped application or packaging concerns.

- `app/src/`
  - Desktop app build logic and packaging helpers.
- `app/android/`, `app/ios/`
  - Mobile-specific builds; usually out of scope unless the task explicitly mentions them.

## `tools/`

This directory is disproportionately useful for debugging because each tool isolates one behavior.

- `tools/Matcher/`
  - Best first stop for SuperPoint/SuperGlue matching validation.
- `tools/CameraRGBD/`
  - Best first stop for basic RGB-D device bring-up including RealSense2.
- `tools/OdometryViewer/`
  - Useful for odometry-specific sensor validation.
- `tools/Calibration/`
  - Useful when camera calibration or raw/rectified assumptions matter.
- `tools/DatabaseViewer/`
  - Good for inspecting parameters and data products after a run.

## `examples/`

Smaller runnable surfaces than the full app, but broader than single-purpose tools.

- `examples/RGBDMapping/`
  - Helpful if the task moves from camera bring-up to actual online mapping behavior.
- Other examples are niche and should be used only when they better match the task.

## `cmake_modules/`

Important when CMake says a backend is missing.

- `FindRealSense2.cmake`
  - Local fallback discovery for librealsense2.
- Similar `Find*.cmake` files exist for many optional SDKs.

## Environment Artifacts

- `.devcontainer/`
  - Several Ubuntu variants plus an "latest deps from source" setup. Good for repeatable agent work.
- `docker/`
  - Broader environment coverage, including ROS-focused images and research snapshots.
- `build/`
  - Existing build tree. Inspect before reuse:
    - `CMakeCache.txt`
    - generated target binaries under `bin/`
    - whether the enabled options match the current task

## Suggested Reading Order By Task

- Build/configuration issue:
  1. `CMakeLists.txt`
  2. relevant `Find*.cmake`
  3. target-level `CMakeLists.txt`

- Feature backend issue:
  1. `Parameters.h`
  2. `Features2d.cpp`
  3. `RegistrationVis.*` or relevant tool

- D455/RealSense2 issue:
  1. `CameraRealSense2.h/.cpp`
  2. `tools/CameraRGBD/main.cpp`
  3. `guilib` preference/about surfaces

- Python matcher issue:
  1. `corelib/src/python/*`
  2. `PythonInterface.h`
  3. `tools/Matcher/main.cpp`
