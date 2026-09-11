#!/bin/bash
#SBATCH --job-name=autostorage-opi-demo
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --gres=lscratch:20

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR}"
RESULTS_DIR="${SUBMIT_DIR}/out"
mkdir -p "${RESULTS_DIR}"

# ---- Set up scratch directory ------------------------------------------
SCRATCH_DIR="/lscratch/${USER}/${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_DIR}"

cp -r "${SUBMIT_DIR}"/. "${SCRATCH_DIR}/"
cd "${SCRATCH_DIR}"

# Copy only out/* back
cleanup() {
    mkdir -p "${RESULTS_DIR}"
    cp -r "${SCRATCH_DIR}"/out/* "${RESULTS_DIR}/" 2>/dev/null || true
    rm -rf "${SCRATCH_DIR}"
}
trap cleanup EXIT

# ---- Environment ---------------------------------------------------------
module purge
module load ORCA/6.1.1-gompi-2023b-avx2

uv run python autostorage_opi_demo/1_GOAT.py