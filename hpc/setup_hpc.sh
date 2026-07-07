#!/bin/bash
# ============================================================
# One-time setup for NYU Torch HPC (conda-based)
# Run ONCE after copying the repo to /scratch/yd3288/
# ============================================================
set -euo pipefail

REPO_DIR="/scratch/yd3288/RoofNet-xAI"
CONDA_ENV_PATH="/scratch/yd3288/conda-envs/roofnet"
SCRATCH_TMP="/scratch/yd3288/tmp"
PIP_CACHE_DIR="/scratch/yd3288/pip-cache"
CONDA_PKGS_DIRS="/scratch/yd3288/conda-pkgs"
CONDA_ENVS_PATH="/scratch/yd3288/conda-envs"
SCRATCH_HOME="/scratch/yd3288/home"
CONDARC="/scratch/yd3288/condarc"
HF_HOME="/scratch/yd3288/hf-cache"

# NYU Torch has small /tmp and home quota. Put conda/pip/build caches on scratch.
# HOME is redirected inside this script because conda may still write ~/.conda
# metadata even when the environment and package cache are on scratch.
export HOME="$SCRATCH_HOME"
export TMPDIR="$SCRATCH_TMP"
export TEMP="$SCRATCH_TMP"
export TMP="$SCRATCH_TMP"
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS"
export CONDA_ENVS_PATH="$CONDA_ENVS_PATH"
export CONDARC="$CONDARC"
export HF_HOME="$HF_HOME"

# NYU Anaconda activation hooks can reference this variable. Keep defined.
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-}"

echo "=== NYU Torch HPC Setup ==="
echo "Repo:       $REPO_DIR"
echo "Env path:   $CONDA_ENV_PATH"
echo "TMPDIR:     $TMPDIR"
echo "Pip cache:  $PIP_CACHE_DIR"
echo "Conda pkgs: $CONDA_PKGS_DIRS"
echo "Conda home: $HOME"
echo "HF cache:   $HF_HOME"
echo ""

mkdir -p "$SCRATCH_TMP" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH" "$SCRATCH_HOME" "$HF_HOME"
cat > "$CONDARC" <<EOF
pkgs_dirs:
  - $CONDA_PKGS_DIRS
envs_dirs:
  - $CONDA_ENVS_PATH
EOF

# --- 1. Load modules ---
# Disable nounset around module/conda hooks because NYU scripts may read unset vars.
set +u
module purge
module load anaconda3/2025.06
eval "$(conda shell.bash hook)"
set -u

# --- 2. Create or reuse conda environment on scratch ---
if [ -d "$CONDA_ENV_PATH" ]; then
    echo ">>> Reusing existing conda environment: $CONDA_ENV_PATH"
else
    echo ">>> Creating conda environment on scratch ..."
    conda create -y -p "$CONDA_ENV_PATH" --override-channels -c conda-forge python=3.11
fi

set +u
conda activate "$CONDA_ENV_PATH"
set -u

# --- 3. Install dependencies ---
echo ">>> Installing xAI requirements ..."
python -m pip install --no-cache-dir --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r "$REPO_DIR/requirements-xai.txt"

# --- 4. Verify ---
echo ""
echo ">>> Verifying imports ..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
import open_clip; print(f'open_clip: {open_clip.__version__}')
import transformers; print(f'transformers: {transformers.__version__}')
from PIL import Image; print('PIL: OK')
import pandas; print(f'pandas: {pandas.__version__}')
import cv2; print(f'opencv: {cv2.__version__}')
print('All imports OK.')
"

# --- 5. Create output directories ---
echo ">>> Creating output directories ..."
mkdir -p "$REPO_DIR/xAI_outputs/segmentation/logs"

echo ""
echo "=== Setup complete ==="
echo "Activate later with:"
echo "  module load anaconda3/2025.06"
echo "  eval \"\$(conda shell.bash hook)\""
echo "  conda activate $CONDA_ENV_PATH"
echo "Next: edit --account in hpc/run_job.sbatch, then: sbatch hpc/run_job.sbatch"
