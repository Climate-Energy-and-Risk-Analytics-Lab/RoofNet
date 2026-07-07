#!/bin/bash
# ============================================================
# Slurm job array for NYU Torch HPC
# RemoteCLIP Segmentation Overlap — 4 parallel workers
#
# Submit:  sbatch hpc/run_job.sbatch
# Monitor: squeue --me
# Cancel:  scancel -u yd3288
# ============================================================
#SBATCH --job-name=seg_overlap
#SBATCH --account=<YOUR_SLURM_ACCOUNT>   # run: my_slurm_accounts
#SBATCH --array=0-3                       # 4 parallel jobs (indices 0,1,2,3)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=04:00:00                   # 4h per worker (was 12h)
#SBATCH --output=/scratch/yd3288/RoofNet-xAI/xAI_outputs/segmentation/logs/%A_%a_out.txt
#SBATCH --error=/scratch/yd3288/RoofNet-xAI/xAI_outputs/segmentation/logs/%A_%a_err.txt

set -euo pipefail

REPO_DIR="/scratch/yd3288/RoofNet-xAI"
CONDA_ENV_PATH="/scratch/yd3288/conda-envs/roofnet"
SCRIPT="$REPO_DIR/xAI_notebooks/remoteclip_segmentation_overlap_batch.py"

# --- Work partitioning ---
# 625 images ÷ 4 workers = 156 images each (last picks up remainder)
CHUNK_SIZE=157
OFFSET=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))

# --- Load environment ---
export HOME="/scratch/yd3288/home"
export TMPDIR="/scratch/yd3288/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export PIP_CACHE_DIR="/scratch/yd3288/pip-cache"
export CONDA_PKGS_DIRS="/scratch/yd3288/conda-pkgs"
export CONDA_ENVS_PATH="/scratch/yd3288/conda-envs"
export CONDARC="/scratch/yd3288/condarc"
export HF_HOME="/scratch/yd3288/hf-cache"
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-}"
mkdir -p "$HOME" "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH" "$HF_HOME"

# Disable nounset around module/conda hooks because NYU scripts may read unset vars.
set +u
module purge
module load anaconda3/2025.06
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PATH"
set -u

# --- Verify GPU ---
echo "=== Job ${SLURM_ARRAY_TASK_ID} started: $(date) ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true
echo "Processing images $OFFSET to $((OFFSET + CHUNK_SIZE - 1))"
echo ""

# --- Pre-flight ---
if [ ! -f "$REPO_DIR/best_clip_model_balanced.pth" ]; then
    echo "ERROR: Missing model weights"
    exit 1
fi
if [ ! -d "$REPO_DIR/RoofNet-Images" ]; then
    echo "ERROR: Missing images"
    exit 1
fi

# --- Run ---
python "$SCRIPT" \
    --split holdout \
    --method transformer_explainability \
    --output-dir "$REPO_DIR/xAI_outputs/segmentation" \
    --model-weights "$REPO_DIR/best_clip_model_balanced.pth" \
    --image-dir "$REPO_DIR/RoofNet-Images" \
    --metadata-csv "$REPO_DIR/roofnet_metadata.csv" \
    --device cuda \
    --log-level INFO \
    --offset "$OFFSET" \
    --limit "$CHUNK_SIZE"

echo ""
echo "=== Job ${SLURM_ARRAY_TASK_ID} finished: $(date) ==="
