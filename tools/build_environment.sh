#!/bin/bash
set -euo pipefail
trap 'STATUS=$?; printf "%s\n" "$STATUS" > logs/build-environment.exit' EXIT
BASE_IMAGE="${1:-${BASE_IMAGE:-}}"
if [ -z "$BASE_IMAGE" ] || [ ! -f "$BASE_IMAGE" ]; then
    printf '%s\n' 'Usage: bash tools/build_environment.sh /absolute/path/to/COLMAP-3.13-base.sif' >&2
    exit 2
fi
apptainer build --build-arg "base_image=$BASE_IMAGE" containers/alignment313.sif containers/alignment313.def
