#!/bin/bash
#SBATCH --job-name=barranco-inventory
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?}"
PIPELINE="${1:?pipeline root}"
SCRATCH="/local/$USER/blockalign-inventory/${SLURM_JOB_ID:?}"
mkdir -p "$SCRATCH"
trap 'rm -rf -- "$SCRATCH"' EXIT
cp -R "$ROOT/src" "$ROOT/tools" "$SCRATCH/"
srun apptainer exec --cleanenv --bind "$SCRATCH:$SCRATCH" \
  --env "PYTHONPATH=$SCRATCH/src" --env "SLURM_JOB_ID=$SLURM_JOB_ID" --env PYTHONUNBUFFERED=1 \
  --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  "$ROOT/containers/alignment313.sif" python3 -B "$SCRATCH/tools/inventory_barranco.py" \
  --pipeline "$PIPELINE" --output "$ROOT/results/barranco-inventory-$SLURM_JOB_ID"
