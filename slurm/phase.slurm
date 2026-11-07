#!/bin/bash
# Resources are supplied explicitly by sbatch CLI, with no implicit GPU request.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?}"
PHASE="${1:?phase}"
SCRATCH="/local/$USER/alignment005009/${SLURM_JOB_ID:?}"
mkdir -p "$SCRATCH"
trap 'rm -rf -- "$SCRATCH"' EXIT
cp -R "$ROOT/src" "$SCRATCH/src"
cp -R "$ROOT/configs" "$SCRATCH/configs"
mkdir -p "$ROOT/results/provenance/$SLURM_JOB_ID"
cp -R "$SCRATCH/src" "$ROOT/results/provenance/$SLURM_JOB_ID/src"
cp -R "$SCRATCH/configs" "$ROOT/results/provenance/$SLURM_JOB_ID/configs"
export APPTAINERENV_ALIGNEXP_ROOT="$ROOT"
export APPTAINERENV_PYTHONPATH="$SCRATCH/src"
OPTS=(--cleanenv --bind "$SCRATCH:$SCRATCH" --env "ALIGNEXP_ROOT=$ROOT" --env "PYTHONPATH=$SCRATCH/src" \
      --env "ALIGNEXP_CONFIG=$SCRATCH/configs/experiment.yaml" \
      --env "TMPDIR=$SCRATCH" --env "SLURM_JOB_ID=$SLURM_JOB_ID" --env "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK" \
      --env PYTHONUNBUFFERED=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OMP_NUM_THREADS=1 --env MPLBACKEND=Agg)
srun apptainer exec "${OPTS[@]}" "$ROOT/containers/alignment313.sif" python3 -B -m alignexp.cli "$PHASE"
