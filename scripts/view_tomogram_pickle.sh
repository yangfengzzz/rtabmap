#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
ros_setup="/opt/ros/jazzy/setup.bash"
rviz_config="${repo_dir}/data/tomogram_view.rviz"
publisher_script="${repo_dir}/scripts/view_tomogram_pickle.py"
venv_python="${repo_dir}/.venv-rtabmap/bin/python"
publisher_log="/tmp/rtabmap_tomogram_viewer_${$}.log"

usage() {
  cat <<'EOF'
Usage:
  scripts/view_tomogram_pickle.sh [tomogram.pickle]

Description:
  Publish a tomogram pickle to ROS 2 PointCloud2 topics and open RViz.

Behavior:
  - If tomogram.pickle is omitted, use the newest .pickle under
    data/tomography/exports, otherwise fall back to
    ~/Desktop/navmap/tomography/exports.
  - If a same-basename .pcd exists next to the pickle, it is also published
    on /global_points to match the upstream PCT viewer flow.
EOF
}

die() {
  echo "$*" >&2
  exit 1
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [[ -n "${rviz_pid:-}" ]]; then
    kill "${rviz_pid}" >/dev/null 2>&1 || true
    wait "${rviz_pid}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${publisher_pid:-}" ]]; then
    kill "${publisher_pid}" >/dev/null 2>&1 || true
    wait "${publisher_pid}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${tf_base_link_pid:-}" ]]; then
    kill "${tf_base_link_pid}" >/dev/null 2>&1 || true
    wait "${tf_base_link_pid}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${tf_base_dash_pid:-}" ]]; then
    kill "${tf_base_dash_pid}" >/dev/null 2>&1 || true
    wait "${tf_base_dash_pid}" >/dev/null 2>&1 || true
  fi
  exit "${code}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

[[ -f "${ros_setup}" ]] || die "ROS 2 setup not found: ${ros_setup}"
[[ -f "${rviz_config}" ]] || die "RViz config not found: ${rviz_config}"
[[ -f "${publisher_script}" ]] || die "Publisher script not found: ${publisher_script}"
[[ -x "${venv_python}" ]] || die "Repo venv python not found: ${venv_python}"

set +u
source "${ros_setup}"
set -u

command -v ros2 >/dev/null 2>&1 || die "ros2 command not found after sourcing ${ros_setup}"
command -v rviz2 >/dev/null 2>&1 || die "rviz2 command not found after sourcing ${ros_setup}"

pickle_path="${1:-}"
if [[ -z "${pickle_path}" ]]; then
  newest="$(find "${repo_dir}/data/tomography/exports" -maxdepth 1 -type f -name '*.pickle' -printf '%T@|%p\n' 2>/dev/null | sort -t'|' -k1,1nr | head -n1 || true)"
  if [[ -z "${newest}" ]]; then
    newest="$(find "${HOME}/Desktop/navmap/tomography/exports" -maxdepth 1 -type f -name '*.pickle' -printf '%T@|%p\n' 2>/dev/null | sort -t'|' -k1,1nr | head -n1 || true)"
  fi
  [[ -n "${newest}" ]] || die "No .pickle found under ${repo_dir}/data/tomography/exports or ~/Desktop/navmap/tomography/exports"
  pickle_path="${newest#*|}"
fi

if [[ "${pickle_path}" == "~/"* ]]; then
  pickle_path="${HOME}/${pickle_path#~/}"
fi
pickle_path="$(cd "$(dirname "${pickle_path}")" && pwd)/$(basename "${pickle_path}")"
[[ -f "${pickle_path}" ]] || die "Tomogram pickle not found: ${pickle_path}"

trap cleanup EXIT INT TERM

echo "Using tomogram pickle: ${pickle_path}"

"${venv_python}" "${publisher_script}" --pickle "${pickle_path}" >"${publisher_log}" 2>&1 &
publisher_pid=$!

for _ in {1..20}; do
  if ! kill -0 "${publisher_pid}" >/dev/null 2>&1; then
    cat "${publisher_log}" >&2 || true
    die "Tomogram publisher failed to start"
  fi
  if ros2 topic list 2>/dev/null | grep -qx '/tomogram'; then
    break
  fi
  sleep 0.5
done

if ! kill -0 "${publisher_pid}" >/dev/null 2>&1; then
  cat "${publisher_log}" >&2 || true
  die "Tomogram publisher failed to start"
fi

ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map base_link >/tmp/rtabmap_tomogram_tf_base_link.log 2>&1 &
tf_base_link_pid=$!
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map base-link >/tmp/rtabmap_tomogram_tf_base_dash.log 2>&1 &
tf_base_dash_pid=$!

echo "Launching RViz..."
rviz2 -d "${rviz_config}" &
rviz_pid=$!

wait "${rviz_pid}"
