#!/bin/bash
# ============================================================
# Cleanup NYU Torch HPC local RoofNet-xAI environment/cache files
# ============================================================
set -euo pipefail

CONDA_ENV_PATH="/scratch/yd3288/conda-envs/roofnet"
SCRATCH_TMP="/scratch/yd3288/tmp"
PIP_CACHE_DIR="/scratch/yd3288/pip-cache"
CONDA_PKGS_DIRS="/scratch/yd3288/conda-pkgs"
CONDA_ENVS_PATH="/scratch/yd3288/conda-envs"
SCRATCH_HOME="/scratch/yd3288/home"
CONDARC="/scratch/yd3288/condarc"
HF_CACHE_DIR="${HF_HOME:-/scratch/yd3288/hf-cache}"

usage() {
    cat <<'EOF'
Usage: bash hpc/cleanup_hpc_env.sh [--yes] [--hf-cache]

Removes:
  - /scratch/yd3288/conda-envs/roofnet
  - /scratch/yd3288/tmp/pip-* and build temp leftovers
  - /scratch/yd3288/pip-cache
  - /scratch/yd3288/conda-pkgs
  - /scratch/yd3288/home/.conda metadata
  - /scratch/yd3288/condarc

Optional:
  --hf-cache   also remove HF cache path shown by script
  --yes        skip confirmation prompt
EOF
}

ASSUME_YES=0
REMOVE_HF_CACHE=0
for arg in "$@"; do
    case "$arg" in
        --yes) ASSUME_YES=1 ;;
        --hf-cache) REMOVE_HF_CACHE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $arg"; usage; exit 2 ;;
    esac
done

echo "=== RoofNet-xAI HPC cleanup ==="
echo "Conda env:  $CONDA_ENV_PATH"
echo "Scratch tmp: $SCRATCH_TMP"
echo "Pip cache:   $PIP_CACHE_DIR"
echo "Conda pkgs:  $CONDA_PKGS_DIRS"
echo "Conda home:  $SCRATCH_HOME"
echo "Condarc:     $CONDARC"
if [ "$REMOVE_HF_CACHE" -eq 1 ]; then
    echo "HF cache:    $HF_CACHE_DIR"
fi
echo ""

if [ "$ASSUME_YES" -ne 1 ]; then
    read -r -p "Delete these files? Type 'yes' to continue: " answer
    if [ "$answer" != "yes" ]; then
        echo "Aborted."
        exit 0
    fi
fi

# Conda may not be loaded; direct rm is fine for path-based scratch env.
rm -rf "$CONDA_ENV_PATH"
rm -rf "$PIP_CACHE_DIR"
rm -rf "$CONDA_PKGS_DIRS"
rm -rf "$SCRATCH_HOME/.conda"
rm -f "$CONDARC"
rm -rf "$SCRATCH_TMP"/pip-* "$SCRATCH_TMP"/build-* "$SCRATCH_TMP"/tmp*

if [ "$REMOVE_HF_CACHE" -eq 1 ]; then
    rm -rf "$HF_CACHE_DIR"
fi

echo "Cleanup complete."
