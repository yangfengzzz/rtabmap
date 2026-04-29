#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
reprocess_bin="${RTABMAP_REPROCESS_BIN:-${repo_dir}/build/bin/rtabmap-reprocess}"
sqlite3_bin="${SQLITE3_BIN:-sqlite3}"
python3_bin="${PYTHON3_BIN:-python3}"
default_db_dir="${HOME}/Documents/RTAB-Map"
occupied_thresh="0.5"
free_thresh="0.196"
clean_intermediate=false

usage() {
  cat <<'EOF'
Usage:
  scripts/export_nav2_map.sh [input.db] [output_dir] [--clean]

Description:
  Rebuild a 2D occupancy map from an RTAB-Map database and export a Nav2 map bundle:
    - map.pgm
    - map.yaml

Behavior:
  - If input.db is omitted, the newest *.db in ~/Documents/RTAB-Map is used.
  - If output_dir is omitted, <db-dir>/<db-basename>_nav2 is used.
  - --clean removes the intermediate rebuilt database after export.
EOF
}

die() {
  echo "$*" >&2
  exit 1
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

export_db_map_to_pgm() {
  local input_db="$1"
  local output_pgm="$2"
  "${python3_bin}" - "$input_db" "$output_pgm" <<'PY'
import sqlite3
import struct
import sys
import zlib

db_path, out_path = sys.argv[1], sys.argv[2]

con = sqlite3.connect(db_path)
row = con.execute("SELECT opt_map FROM Admin LIMIT 1;").fetchone()
con.close()

if not row or row[0] is None:
    raise RuntimeError(f"No opt_map blob found in {db_path}")

blob = row[0]
if len(blob) < 12:
    raise RuntimeError(f"opt_map blob too small in {db_path}")

rows, cols, cv_type = struct.unpack("<iii", blob[-12:])
if cv_type != 1:
    raise RuntimeError(f"Unsupported OpenCV map type {cv_type} in {db_path}; expected CV_8SC1 (1)")

raw = zlib.decompress(blob[:-12])
expected = rows * cols
if len(raw) != expected:
    raise RuntimeError(f"Unexpected uncompressed map size {len(raw)} (expected {expected})")

# Match RTAB-Map util3d::convertMap2Image8U(map, true):
# - vertical flip for PGM
# - occupancy 0 -> 254, 100 -> 0, -2 -> 254, everything else -> 205
out = bytearray(expected)
for i in range(rows):
    src_row = rows - 1 - i
    src_base = src_row * cols
    dst_base = i * cols
    for j in range(cols):
        v = raw[src_base + j]
        if v > 127:
            v -= 256
        if v == 0:
            gray = 254
        elif v == 100:
            gray = 0
        elif v == -2:
            gray = 254
        else:
            gray = 205
        out[dst_base + j] = gray

with open(out_path, "wb") as f:
    f.write(f"P5\n{cols} {rows}\n255\n".encode("ascii"))
    f.write(out)
PY
}

input_db=""
output_dir=""

for arg in "$@"; do
  case "$arg" in
    --help|-h)
      usage
      exit 0
      ;;
    --clean)
      clean_intermediate=true
      ;;
    *)
      if [[ -z "${input_db}" ]]; then
        input_db="${arg}"
      elif [[ -z "${output_dir}" ]]; then
        output_dir="${arg}"
      else
        die "Unexpected argument: ${arg}"
      fi
      ;;
  esac
done

if [[ ! -x "${reprocess_bin}" ]]; then
  die "Missing required binary: ${reprocess_bin}"
fi

if ! have_cmd "${sqlite3_bin}"; then
  die "Missing required command: ${sqlite3_bin}"
fi

if ! have_cmd "${python3_bin}"; then
  die "Missing required command: ${python3_bin}"
fi

if [[ -z "${input_db}" ]]; then
  [[ -d "${default_db_dir}" ]] || die "Default database directory not found: ${default_db_dir}"
  newest_entry="$(find "${default_db_dir}" -maxdepth 1 -type f -name '*.db' -printf '%T@|%p\n' | sort -t'|' -k1,1nr | head -n1 || true)"
  [[ -n "${newest_entry}" ]] || die "No .db files found in ${default_db_dir}"
  input_db="${newest_entry#*|}"
fi

if [[ "${input_db}" == "~/"* ]]; then
  input_db="${HOME}/${input_db#~/}"
fi

[[ -f "${input_db}" ]] || die "Database not found: ${input_db}"
[[ "${input_db}" == *.db ]] || die "Input database must end with .db: ${input_db}"

input_db="$(cd "$(dirname "${input_db}")" && pwd)/$(basename "${input_db}")"
db_dir="$(dirname "${input_db}")"
db_name="$(basename "${input_db}" .db)"

if [[ -z "${output_dir}" ]]; then
  output_dir="${db_dir}/${db_name}_nav2"
fi

mkdir -p "${output_dir}"
output_dir="$(cd "${output_dir}" && pwd)"

intermediate_db="${output_dir}/${db_name}_rebuilt.db"
final_pgm="${output_dir}/map.pgm"
final_yaml="${output_dir}/map.yaml"

echo "Selected database: ${input_db}"
echo "Output directory: ${output_dir}"

rm -f "${intermediate_db}" "${output_dir}/${db_name}_rebuilt_map.pgm" "${final_pgm}" "${final_yaml}"

echo "Rebuilding 2D occupancy map metadata..."
"${reprocess_bin}" -g2 -db --RGBD/CreateOccupancyGrid true "${input_db}" "${intermediate_db}"

[[ -f "${intermediate_db}" ]] || die "Expected rebuilt database was not generated: ${intermediate_db}"

metadata="$("${sqlite3_bin}" -separator '|' "${intermediate_db}" "SELECT opt_map_x_min, opt_map_y_min, opt_map_resolution FROM Admin LIMIT 1;" || true)"
[[ -n "${metadata}" ]] || die "Failed to read map metadata from rebuilt database: ${intermediate_db}"

IFS='|' read -r x_min y_min resolution <<< "${metadata}"

[[ -n "${x_min}" ]] || die "Map metadata opt_map_x_min is empty in ${intermediate_db}"
[[ -n "${y_min}" ]] || die "Map metadata opt_map_y_min is empty in ${intermediate_db}"
[[ -n "${resolution}" ]] || die "Map metadata opt_map_resolution is empty in ${intermediate_db}"

echo "Exporting PGM from rebuilt database map blob..."
export_db_map_to_pgm "${intermediate_db}" "${final_pgm}"
[[ -f "${final_pgm}" ]] || die "Expected final map image was not generated: ${final_pgm}"

cat > "${final_yaml}" <<EOF
image: map.pgm
resolution: ${resolution}
origin: [${x_min}, ${y_min}, 0.0]
negate: 0
occupied_thresh: ${occupied_thresh}
free_thresh: ${free_thresh}
EOF

if [[ "${clean_intermediate}" == true ]]; then
  rm -f "${intermediate_db}" "${output_dir}/${db_name}_rebuilt_map.pgm"
fi

echo "Generated Nav2 map image: ${final_pgm}"
echo "Generated Nav2 map yaml:  ${final_yaml}"
