#!/bin/bash
set -euo pipefail
trap 'STATUS=$?; printf "%s\n" "$STATUS" > logs/build-environment.exit' EXIT
apptainer build containers/alignment313.sif containers/alignment313.def
