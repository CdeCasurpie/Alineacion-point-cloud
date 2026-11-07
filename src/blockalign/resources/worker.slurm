#!/bin/bash
set -euo pipefail
ROOT="${1:?workspace}"
RUN="${2:?run id}"
IMAGE="${3:?container}"
SCRATCH="/local/$USER/blockalign/${SLURM_JOB_ID:?}"
mkdir -p "$SCRATCH"
trap 'rm -rf -- "$SCRATCH"' EXIT
cp -R "$ROOT/runs/$RUN/src" "$SCRATCH/src"
srun apptainer exec --cleanenv --bind "$SCRATCH:$SCRATCH" \
  --env "PYTHONPATH=$SCRATCH/src" --env "TMPDIR=$SCRATCH" \
  --env "SLURM_JOB_ID=$SLURM_JOB_ID" --env "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK" \
  --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env MPLBACKEND=Agg \
  "$IMAGE" python3 -B -m blockalign.worker "$ROOT" "$RUN"
