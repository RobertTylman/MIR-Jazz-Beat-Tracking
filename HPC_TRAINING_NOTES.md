# HPC Training Notes (TORCH)

## Why full-dataset eval is biased
If you train on JTD and then evaluate on all JTD tracks, results are optimistic.
Use a strict split:
- `train`: fit model weights
- `val`: tune hyperparameters / early stopping
- `test`: final reporting only (never touched during tuning)

## Recommended protocol
1. Build dataset once into Beat This format.
2. Create and freeze a `train/val/test` split.
3. Train on `train` with early stopping on `val`.
4. Select best checkpoint by validation metric.
5. Evaluate once on `test`.
6. Compare baseline and finetuned models on the exact same test set.

## Current paths used
- Processed dataset: `/scratch/rht9410/beatthis_jtd`
- Best finetuned checkpoint: `/scratch/rht9410/checkpoints/jtd_finetune/jtd-ft-05-0.9793.ckpt`
- Full eval output: `/scratch/rht9410/eval_outputs/beat_this_jtd_finetuned.csv`

## GPU allocation (working)
```bash
sbatch --account=torch_pr_1003_general --partition=l40s_public --cpus-per-task=8 --gres=gpu:1 --mem=32G --time=08:00:00 --wrap "sleep infinity"
squeue -u $USER
srun --jobid=<JOBID> --pty /bin/bash
```

## Environment check command
```bash
PY=/scratch/rht9410/envs/jazzft/bin/python
$PY - << 'PY'
import torch, torchaudio, numpy, scipy, pytorch_lightning
print(torch.__version__, torch.cuda.is_available(), torchaudio.__version__)
print(numpy.__version__, scipy.__version__, pytorch_lightning.__version__)
PY
```

## Finetune command (JTD)
```bash
PY=/scratch/rht9410/envs/jazzft/bin/python
cd ~/MIR-Jazz-Beat-Tracking
PYTHONPATH=/home/rht9410/MIR-Jazz-Beat-Tracking $PY scripts/finetune_beat_this_jtd.py \
  --data-dir /scratch/rht9410/beatthis_jtd \
  --checkpoint beat_this/checkpoint/final0.ckpt \
  --output-dir /scratch/rht9410/checkpoints/jtd_finetune \
  --accelerator gpu --devices 1 --precision 16-mixed \
  --batch-size 8 --num-workers 4 --max-epochs 40 --lr 2e-4 \
  --wandb-project mir-jazz-beat-tracking --wandb-run-name torch-jtd-ft
```

## Evaluation command (write to scratch)
```bash
PY=/scratch/rht9410/envs/jazzft/bin/python
cd ~/MIR-Jazz-Beat-Tracking
PYTHONPATH=/home/rht9410/MIR-Jazz-Beat-Tracking $PY scripts/evaluate_jtd.py \
  --data-root /scratch/rht9410/jazz-trio-database-v02 \
  --audio-root /home/rht9410/jtd_data \
  --checkpoint /scratch/rht9410/checkpoints/jtd_finetune/jtd-ft-05-0.9793.ckpt \
  --output /scratch/rht9410/eval_outputs/beat_this_jtd_finetuned.csv \
  --device cuda
```

## W&B quota-safe settings
```bash
export WANDB_DIR=/scratch/rht9410/wandb
export WANDB_DATA_DIR=/scratch/rht9410/wandb-data
export WANDB_CACHE_DIR=/scratch/rht9410/wandb-cache
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR"
```

## NFS cleanup warnings
Messages like `.nfs... Device or resource busy` are common on network filesystems during multiprocessing cleanup.
They are usually non-fatal if training/eval outputs are still produced.
