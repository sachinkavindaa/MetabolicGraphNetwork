#!/bin/bash -l
#SBATCH --job-name=ko2ec_rxn
#SBATCH --partition=guest,batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=ko2ec_rxn.%j.out
#SBATCH --error=ko2ec_rxn.%j.err
## Optional email:
## #SBATCH --mail-user=echandrasekara2@unl.edu
## #SBATCH --mail-type=END,FAIL

set -euo pipefail

# --- Hygiene: keep threaded libs polite on shared nodes ---
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# --- Paths (edit if you want) ---
INPUT_TXT="/work/samodha/sachin/MWAS/MetabolicGraphNetwork/full_network/Full_data.txt"
OUT_CSV="/work/samodha/sachin/MWAS/MetabolicGraphNetwork/full_network/Full_data_EX_RXN.csv"
PY_SCRIPT="/work/samodha/sachin/MWAS/MetabolicGraphNetwork/full_network/01_EC_Reaction.py"  # <-- put your Python code here

# --- Python environment on HCC ---
module purge
# If HCC has a python module you use, load it. Otherwise comment this out and rely on system python.
module load python/3.10

# Use a throwaway venv in node-local scratch
VENV_DIR="${SLURM_TMPDIR:-/tmp}/venv_ko2ec_rxn"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir pandas

echo "[INFO] Python: $(python --version)"
echo "[INFO] Using venv at: $VENV_DIR"

# --- Run job ---
set -x
python "$PY_SCRIPT"  # your script uses hardcoded paths already
set +x

echo "[INFO] Done. Output should be at: $OUT_CSV"
