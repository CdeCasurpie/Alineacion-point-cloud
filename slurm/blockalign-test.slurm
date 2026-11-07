#!/bin/bash
#SBATCH --job-name=blockalign-tests
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
SCRATCH="/local/$USER/blockalign-tests/${SLURM_JOB_ID:?}"
mkdir -p "$SCRATCH"
trap 'rm -rf -- "$SCRATCH"' EXIT
cp -R src tests "$SCRATCH/"
srun apptainer exec --cleanenv --bind "$SCRATCH:$SCRATCH" \
  --env "PYTHONPATH=$SCRATCH/src" --env "TMPDIR=$SCRATCH" --env "SLURM_JOB_ID=$SLURM_JOB_ID" \
  --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MPLBACKEND=Agg \
  containers/alignment313.sif python3 -B -m pytest -q -p no:cacheprovider "$SCRATCH/tests"
