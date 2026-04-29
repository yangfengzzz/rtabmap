#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
preset_path="${repo_dir}/data/presets/d455_default_baseline.ini"
config_dir="${HOME}/.rtabmap"
config_path="${config_dir}/rtabmap.ini"
torch_lib_dir="${repo_dir}/.venv-rtabmap/lib/python3.12/site-packages/torch/lib"

if [[ ! -x "${repo_dir}/build/bin/rtabmap" ]]; then
  echo "Missing GUI binary: ${repo_dir}/build/bin/rtabmap" >&2
  exit 1
fi

if [[ ! -f "${preset_path}" ]]; then
  echo "Missing preset: ${preset_path}" >&2
  exit 1
fi

mkdir -p "${config_dir}"

if [[ -f "${config_path}" ]] && ! cmp -s "${preset_path}" "${config_path}"; then
  backup_path="${config_path}.backup.$(date +%Y%m%d-%H%M%S)"
  cp "${config_path}" "${backup_path}"
  echo "Backed up existing RTAB-Map config to ${backup_path}"
fi

cp "${preset_path}" "${config_path}"

export PATH="/usr/local/cuda/bin:${PATH}"

if [[ -d /usr/local/cuda/lib64 ]] && [[ -d "${torch_lib_dir}" ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${torch_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
elif [[ -d /usr/local/cuda/lib64 ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
elif [[ -d "${torch_lib_dir}" ]]; then
  export LD_LIBRARY_PATH="${torch_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "Launching RTAB-Map with preset ${preset_path}"
echo "Config installed at ${config_path}"

exec "${repo_dir}/build/bin/rtabmap" "$@"
