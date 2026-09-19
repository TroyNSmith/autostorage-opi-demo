#!/bin/bash
#SBATCH --job-name=autostorage-opi-demo
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --gres=lscratch:20

NTASKS=8
MEMORY=8

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
    # sleep 15 # Make sure the program is complete
    mkdir -p "${RESULTS_DIR}"
    cp -r "${SCRATCH_DIR}"/out/* "${RESULTS_DIR}/" 2>/dev/null || true
    rm -rf "${SCRATCH_DIR}"
}
trap cleanup EXIT

# ---- Environment ---------------------------------------------------------
module purge
module load ORCA/6.1.1-gompi-2023b-avx2

# $(which orca) out/3_SCAN/scan/scan.inp > "${SUBMIT_DIR}"/hf_scan.log

#uv run python autostorage_opi_demo/1_GOAT.py -m "$MEMORY" -n "$NTASKS"
#uv run python autostorage_opi_demo/2_OPT.py -m "$MEMORY" -n "$NTASKS"
#uv run python autostorage_opi_demo/3_SCAN.py -m "$MEMORY" -n "$NTASKS"
#uv run python autostorage_opi_demo/4_NEB.py -m "$MEMORY" -n "$NTASKS"
uv run python autostorage_opi_demo/5_IRC.py -m "$MEMORY" -n "$NTASKS"