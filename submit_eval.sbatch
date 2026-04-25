#!/bin/bash
#SBATCH --job-name=jtd-beat-this
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tylman.robbie@gmail.com
#
# Submit:  sbatch submit_eval.sbatch
# Resume after timeout: just sbatch again — evaluate_jtd.py skips finished tracks.

set -euo pipefail

# --- paths --------------------------------------------------------------------
REPO=/scratch/$USER/MIR-Jazz-Beat-Tracking      # <-- adjust if your repo lives elsewhere
DATA_ROOT=$REPO/jazz-trio-database-v02
AUDIO_ROOT=/scratch/$USER/jtd_data
CHECKPOINT=$REPO/beat_this/checkpoint/final0.ckpt
OUTPUT=$REPO/results/beat_this_jtd.csv
PREDS_DIR=$REPO/results/preds                   # remove --save-preds-dir below if not wanted

mkdir -p "$REPO/logs" "$(dirname "$OUTPUT")" "$PREDS_DIR"

# --- environment --------------------------------------------------------------
# Greene-style env activation. Replace the next two lines with whatever you use
# (Singularity overlay, conda, venv, etc.).
module purge
module load anaconda3/2024.06
source activate beatthis    # <-- name of your conda env with beat_this + mir_eval + pandas

# Sanity: print GPU + versions
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# --- run ----------------------------------------------------------------------
cd "$REPO"

srun python evaluate_jtd.py \
    --data-root  "$DATA_ROOT" \
    --audio-root "$AUDIO_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --output     "$OUTPUT" \
    --device     cuda \
    --float16 \
    --save-preds-dir "$PREDS_DIR"

echo "Job finished. Results: $OUTPUT"
