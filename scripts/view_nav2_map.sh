#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
ros_setup="/opt/ros/jazzy/setup.bash"
rviz_config="${repo_dir}/data/nav2_map_view.rviz"
node_name="rtabmap_nav2_map_server_${$}"
node_target="/${node_name}"
map_server_log="/tmp/${node_name}.log"
lifecycle_log="/tmp/${node_name}.lifecycle.log"

usage() {
  cat <<'EOF'
Usage:
  scripts/view_nav2_map.sh [map.yaml]

Description:
  Launch Nav2 map_server for a map yaml and open RViz configured to display /map.

Behavior:
  - If map.yaml is omitted, use the newest map.yaml under ~/Desktop/navmap, otherwise
    fall back to the newest map.yaml under ~/Documents/RTAB-Map.
  - The script configures and activates its own uniquely named map_server node automatically.
  - Ctrl-C stops both RViz and map_server.
EOF
}

die() {
  echo "$*" >&2
  exit 1
}

get_lifecycle_state() {
  local output
  output="$(ros2 lifecycle get "${node_target}" 2>/dev/null | tail -n 1 | tr -d '\r')"
  if [[ "${output}" == *" is "* ]]; then
    printf '%s\n' "${output##* is }"
  else
    printf '%s\n' "${output}"
  fi
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [[ -n "${rviz_pid:-}" ]]; then
    kill "${rviz_pid}" >/dev/null 2>&1 || true
    wait "${rviz_pid}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${map_server_pid:-}" ]]; then
    kill "${map_server_pid}" >/dev/null 2>&1 || true
    wait "${map_server_pid}" >/dev/null 2>&1 || true
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

set +u
source "${ros_setup}"
set -u

command -v ros2 >/dev/null 2>&1 || die "ros2 command not found after sourcing ${ros_setup}"
command -v rviz2 >/dev/null 2>&1 || die "rviz2 command not found after sourcing ${ros_setup}"

map_yaml="${1:-}"

if [[ -z "${map_yaml}" ]]; then
  newest="$(find "${HOME}/Desktop/navmap" -maxdepth 1 -type f -name 'map.yaml' -printf '%T@|%p\n' 2>/dev/null | sort -t'|' -k1,1nr | head -n1 || true)"
  if [[ -z "${newest}" ]]; then
    newest="$(find "${HOME}/Documents/RTAB-Map" -type f -name 'map.yaml' -printf '%T@|%p\n' 2>/dev/null | sort -t'|' -k1,1nr | head -n1 || true)"
  fi
  [[ -n "${newest}" ]] || die "No map.yaml found under ~/Desktop/navmap or ~/Documents/RTAB-Map"
  map_yaml="${newest#*|}"
fi

if [[ "${map_yaml}" == "~/"* ]]; then
  map_yaml="${HOME}/${map_yaml#~/}"
fi

[[ -f "${map_yaml}" ]] || die "Map yaml not found: ${map_yaml}"
[[ "${map_yaml}" == *.yaml ]] || die "Expected a .yaml map file: ${map_yaml}"

map_yaml="$(cd "$(dirname "${map_yaml}")" && pwd)/$(basename "${map_yaml}")"

echo "Using map yaml: ${map_yaml}"

trap cleanup EXIT INT TERM

ros2 run nav2_map_server map_server --ros-args -r __node:="${node_name}" -p yaml_filename:="${map_yaml}" >"${map_server_log}" 2>&1 &
map_server_pid=$!

for _ in {1..20}; do
  if ! kill -0 "${map_server_pid}" >/dev/null 2>&1; then
    cat "${map_server_log}" >&2 || true
    die "map_server failed to start"
  fi
  if ros2 lifecycle get "${node_target}" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if ! kill -0 "${map_server_pid}" >/dev/null 2>&1; then
  cat "${map_server_log}" >&2 || true
  die "map_server failed to start"
fi

state="$(get_lifecycle_state)"
if [[ "${state}" == "unconfigured [1]" ]]; then
  ros2 lifecycle set "${node_target}" configure >"${lifecycle_log}" 2>&1 || {
    cat "${lifecycle_log}" >&2 || true
    cat "${map_server_log}" >&2 || true
    die "Failed to configure ${node_target}"
  }
  state="$(get_lifecycle_state)"
fi

if [[ "${state}" == "inactive [2]" ]]; then
  ros2 lifecycle set "${node_target}" activate >"${lifecycle_log}" 2>&1 || {
    cat "${lifecycle_log}" >&2 || true
    cat "${map_server_log}" >&2 || true
    die "Failed to activate ${node_target}"
  }
  state="$(get_lifecycle_state)"
fi

[[ "${state}" == "active [3]" ]] || {
  cat "${map_server_log}" >&2 || true
  die "${node_target} did not reach active state, current state: ${state:-unknown}"
}

ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map base_link >/tmp/rtabmap_nav2_tf_base_link.log 2>&1 &
tf_base_link_pid=$!
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map base-link >/tmp/rtabmap_nav2_tf_base_dash.log 2>&1 &
tf_base_dash_pid=$!

echo "${node_target} is active. Launching RViz..."
rviz2 -d "${rviz_config}" &
rviz_pid=$!

wait "${rviz_pid}"
