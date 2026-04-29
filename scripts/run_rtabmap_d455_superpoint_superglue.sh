#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
preset_path="${repo_dir}/data/presets/d455_superpoint_superglue.ini"
config_dir="${HOME}/.rtabmap"
config_path="${config_dir}/rtabmap.ini"
venv_python="${repo_dir}/.venv-rtabmap/bin/python"
convert_script="${repo_dir}/scripts/convert_superglue_weights.py"
raw_superglue="${repo_dir}/.runtime-assets/superglue/SuperGluePretrainedNetwork/models/weights/superglue_indoor.pth"
native_superglue="${repo_dir}/.runtime-assets/superglue/superglue_indoor_native.pth"

if [[ ! -x "${repo_dir}/build/bin/rtabmap" ]]; then
  echo "Missing GUI binary: ${repo_dir}/build/bin/rtabmap" >&2
  exit 1
fi

if [[ ! -f "${preset_path}" ]]; then
  echo "Missing preset: ${preset_path}" >&2
  exit 1
fi

if [[ ! -x "${venv_python}" ]]; then
  echo "Missing helper Python environment: ${venv_python}" >&2
  exit 1
fi

if [[ ! -f "${native_superglue}" ]]; then
  if [[ ! -f "${raw_superglue}" ]]; then
    echo "Missing SuperGlue checkpoint: ${raw_superglue}" >&2
    exit 1
  fi
  echo "Converting SuperGlue checkpoint for native C++ runtime..."
  "${venv_python}" "${convert_script}" "${raw_superglue}" "${native_superglue}"
fi

mkdir -p "${config_dir}"

if [[ -f "${config_path}" ]] && ! cmp -s "${preset_path}" "${config_path}"; then
  backup_path="${config_path}.backup.$(date +%Y%m%d-%H%M%S)"
  cp "${config_path}" "${backup_path}"
  echo "Backed up existing RTAB-Map config to ${backup_path}"
fi

cp "${preset_path}" "${config_path}"

export PATH="${repo_dir}/.venv-rtabmap/bin:/usr/local/cuda/bin:${PATH}"

if [[ -d /usr/local/cuda/lib64 ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${repo_dir}/.venv-rtabmap/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
  export LD_LIBRARY_PATH="${repo_dir}/.venv-rtabmap/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "Launching RTAB-Map with preset ${preset_path}"
echo "Config installed at ${config_path}"

exec "${repo_dir}/build/bin/rtabmap" "$@"
