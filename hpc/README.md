# NYU Torch HPC Runbook

Run RemoteCLIP segmentation-overlap batch analysis on NYU Torch HPC.

## 1. Sync repo to HPC

From local repo root:

```bash
rsync -av \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.DS_Store' \
  ./ dtn.torch.hpc.nyu.edu:/scratch/yd3288/RoofNet-xAI/
```

If DTN fails, use login alias:

```bash
rsync -av \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.DS_Store' \
  ./ torch:/scratch/yd3288/RoofNet-xAI/
```

## 2. SSH and enter repo

```bash
ssh torch
cd /scratch/yd3288/RoofNet-xAI
```

## 3. Verify required assets

```bash
ls -lh best_clip_model_balanced.pth roofnet_metadata.csv
ls RoofNet-Images | head
```

Required:

```text
best_clip_model_balanced.pth
RoofNet-Images/
roofnet_metadata.csv
```

## 4. Run one-time setup

```bash
bash hpc/setup_hpc.sh
```

Setup creates/reuses:

```text
/scratch/yd3288/conda-envs/roofnet
/scratch/yd3288/tmp
/scratch/yd3288/pip-cache
/scratch/yd3288/conda-pkgs
/scratch/yd3288/conda-envs
/scratch/yd3288/home
/scratch/yd3288/hf-cache
/scratch/yd3288/condarc
```

Why scratch: Torch `/tmp` is tiny and home quota can be small; pip/conda installs can fail with `OSError(28, 'No space left on device')` or `Disk quota exceeded`. The setup redirects `HOME`, conda package/env dirs, pip cache, temp files, and HF cache to scratch inside the script.

## 5. Activate env manually, if needed

```bash
export HOME=/scratch/yd3288/home
export CONDARC=/scratch/yd3288/condarc
export CONDA_PKGS_DIRS=/scratch/yd3288/conda-pkgs
export CONDA_ENVS_PATH=/scratch/yd3288/conda-envs
export HF_HOME=/scratch/yd3288/hf-cache
module purge
module load anaconda3/2025.06
eval "$(conda shell.bash hook)"
conda activate /scratch/yd3288/conda-envs/roofnet
```

## 6. Run local smoke test on HPC

For GPU smoke test, use an interactive GPU shell instead of login node:

```bash
srun --account=<YOUR_SLURM_ACCOUNT> --gres=gpu:1 --cpus-per-task=4 --mem=32GB --time=01:00:00 --pty bash
```

Then:

```bash
cd /scratch/yd3288/RoofNet-xAI
export HOME=/scratch/yd3288/home
export CONDARC=/scratch/yd3288/condarc
export CONDA_PKGS_DIRS=/scratch/yd3288/conda-pkgs
export CONDA_ENVS_PATH=/scratch/yd3288/conda-envs
export HF_HOME=/scratch/yd3288/hf-cache
module purge
module load anaconda3/2025.06
eval "$(conda shell.bash hook)"
conda activate /scratch/yd3288/conda-envs/roofnet

python xAI_notebooks/remoteclip_segmentation_overlap_batch.py \
  --limit 2 \
  --output-dir xAI_outputs/segmentation_smoke \
  --device cuda \
  --log-level INFO
```

Inspect smoke outputs:

```bash
find xAI_outputs/segmentation_smoke -maxdepth 2 -type f | sort
cat xAI_outputs/segmentation_smoke/tables/segmentation_overlap_failures.jsonl 2>/dev/null || true
```

## 7. Configure Slurm account

Find account:

```bash
my_slurm_accounts
```

Edit:

```bash
nano hpc/run_job.sbatch
```

Set:

```bash
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
```

Also verify hardcoded paths if NetID/repo path changes:

```bash
REPO_DIR="/scratch/yd3288/RoofNet-xAI"
CONDA_ENV_PATH="/scratch/yd3288/conda-envs/roofnet"
```

## 8. Submit full batch job

```bash
sbatch hpc/run_job.sbatch
```

Current job uses 4 workers:

```bash
#SBATCH --array=0-3
```

Each worker processes a chunk with:

```bash
--offset "$OFFSET" --limit "$CHUNK_SIZE"
```

## 9. Monitor job

```bash
squeue --me
```

Tail logs:

```bash
tail -f xAI_outputs/segmentation/logs/*_out.txt
```

Errors:

```bash
tail -f xAI_outputs/segmentation/logs/*_err.txt
```

Cancel all own jobs if needed:

```bash
scancel -u yd3288
```

## 10. Outputs

Main output root:

```text
xAI_outputs/segmentation/
```

Generated files:

```text
tables/segmentation_overlap_results.csv
tables/segmentation_overlap_failures.jsonl
summary/segmentation_overlap_summary.json
plots/*.png
masks/*__combined_mask.png
overlays/*__segmentation_overlay.png
logs/*.log
```

## 11. Resume or recompute

Resume is automatic. Rerun same job/script; completed images are skipped when CSV row + mask + overlay exist.

Force recompute manually:

```bash
python xAI_notebooks/remoteclip_segmentation_overlap_batch.py \
  --split holdout \
  --method transformer_explainability \
  --output-dir xAI_outputs/segmentation \
  --model-weights best_clip_model_balanced.pth \
  --image-dir RoofNet-Images \
  --metadata-csv roofnet_metadata.csv \
  --device cuda \
  --force-recompute \
  --log-level INFO
```

## 12. Cleanup environment/cache

Prompt before delete:

```bash
bash hpc/cleanup_hpc_env.sh
```

No prompt:

```bash
bash hpc/cleanup_hpc_env.sh --yes
```

Also remove HF cache:

```bash
bash hpc/cleanup_hpc_env.sh --yes --hf-cache
```

## Troubleshooting

### `conda: command not found`

Load Anaconda first:

```bash
module purge
module load anaconda3/2025.06
eval "$(conda shell.bash hook)"
```

### `QT_XCB_GL_INTEGRATION: unbound variable`

Scripts already avoid this by disabling `nounset` around module/conda hooks. If running manually:

```bash
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-}"
```

### `OSError(28, 'No space left on device')`

Use scratch tmp/cache:

```bash
mkdir -p /scratch/yd3288/tmp /scratch/yd3288/pip-cache /scratch/yd3288/conda-pkgs /scratch/yd3288/conda-envs /scratch/yd3288/home /scratch/yd3288/hf-cache
export HOME=/scratch/yd3288/home
export TMPDIR=/scratch/yd3288/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
export PIP_CACHE_DIR=/scratch/yd3288/pip-cache
export CONDA_PKGS_DIRS=/scratch/yd3288/conda-pkgs
export CONDA_ENVS_PATH=/scratch/yd3288/conda-envs
export CONDARC=/scratch/yd3288/condarc
export HF_HOME=/scratch/yd3288/hf-cache
```

### Missing dependencies after failed setup

Clean and rerun setup:

```bash
bash hpc/cleanup_hpc_env.sh --yes
bash hpc/setup_hpc.sh
```
