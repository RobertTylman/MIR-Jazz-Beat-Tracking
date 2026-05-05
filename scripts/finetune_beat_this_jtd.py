#!/usr/bin/env python3
"""Fine-tune Beat This! on a BeatDataModule-formatted JTD dataset.

Training flow (transfer learning):
1) Build BeatDataModule from precomputed spectrogram dataset + split file.
2) Compute class-balancing positive weights from TRAIN items only.
3) Load pretrained Beat This! checkpoint (final0.ckpt by default).
4) Continue optimization on JTD train split, validate on JTD val split.
5) Save best checkpoint by validation beat F-measure; stop early on plateau.

Important:
- This script does NOT evaluate the held-out test split.
- Test evaluation should be run separately with evaluate_jtd.py and test_ids.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from beat_this.dataset.dataset import BeatDataModule
from beat_this.model.pl_module import PLBeatThis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Root directory produced by build_jtd_beatthis_dataset.py. Contains:
    # - annotations/<dataset-name>/single.split
    # - audio/spectrograms/<dataset-name>/*/track.npy
    # - metadata needed by BeatDataModule.
    p.add_argument("--data-dir", type=Path, required=True, help="Processed dataset root.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="beat_this/checkpoint/final0.ckpt",
        # This is the *starting point* for transfer learning.
        # We do NOT train from scratch unless you pass a randomly initialized ckpt.
        help="Pretrained Beat This! checkpoint to initialize from.",
    )
    # Where Lightning writes checkpoints + logs.
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/jtd_finetune"))
    # Data loading / batching knobs.
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    # Number of frames per training crop (frame rate is 50 fps below).
    # 1500 frames ~= 30 seconds.
    p.add_argument("--train-length", type=int, default=1500)
    # Optimizer hyperparameters for finetuning.
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    # Upper bound on training duration. Early stopping can stop earlier.
    p.add_argument("--max-epochs", type=int, default=40)
    # Early stopping patience in validation epochs (monitor = val_F-measure_beat).
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--devices", default="1", help="Lightning devices value, e.g. 1, '0,1', or 'auto'.")
    p.add_argument("--accelerator", default="auto", help="cpu|gpu|tpu|auto")
    p.add_argument("--precision", default="16-mixed", help="Lightning precision string.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dataset-name", default="jtd", help="Dataset name used under annotations/<name>.")
    p.add_argument(
        "--save-top-k",
        type=int,
        default=1,
        help="Number of best checkpoints (by val_F-measure_beat) to keep.",
    )
    p.add_argument("--wandb-project", default=None, help="W&B project name. If unset, disables W&B logging.")
    p.add_argument("--wandb-run-name", default=None, help="Optional W&B run name.")
    return p.parse_args()


def parse_devices(raw: str):
    # Convenience parser so CLI can accept:
    # --devices auto
    # --devices 1
    # --devices 0,1
    if raw == "auto":
        return "auto"
    if "," in raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    try:
        return int(raw)
    except ValueError:
        return raw


def main() -> int:
    args = parse_args()
    # Reproducibility:
    # - Seeds PyTorch, NumPy, Python RNG through Lightning utility.
    # - workers=True also seeds DataLoader workers deterministically.
    pl.seed_everything(args.seed, workers=True)

    if not args.data_dir.is_dir():
        raise SystemExit(f"--data-dir not found: {args.data_dir}")

    # Build datamodule for TRAIN/VAL only.
    #
    # Why test_dataset="__none__":
    # BeatDataModule treats test_dataset as excluded from training split creation.
    # In our JTD-only setup, setting a fake non-matching dataset avoids dropping JTD.
    #
    # Split source:
    # BeatDataModule reads annotations/<dataset-name>/single.split under --data-dir.
    # That file defines which track ids are train vs validation.
    dm = BeatDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_length=args.train_length,
        spect_fps=50,
        test_dataset="__none__",
    )

    # Instantiate train/val datasets + dataloaders for the fit stage.
    dm.setup("fit")
    if not dm.train_items:
        raise SystemExit(
            "No training items found. Check annotations/<dataset>/single.split and data-dir layout."
        )

    # Compute positive class weights from TRAIN split only.
    # Beat/downbeat targets are sparse; this balances BCE losses for rare positives.
    pos_weights = dm.get_train_positive_weights()

    # Core transfer-learning step:
    # load pretrained weights and continue optimization on JTD.
    #
    # strict=False is intentional to tolerate small checkpoint/module key mismatches
    # across Lightning/package versions while still loading compatible weights.
    # THIS LOADS THE CHECKPOINT FILE AND BEAT THIS MODEL FOR FURTHER TRAINING
    model = PLBeatThis.load_from_checkpoint(
        args.checkpoint,
        pos_weights=pos_weights,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        strict=False,
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save model checkpoints ranked by validation beat F-measure.
    # Filename keeps epoch + monitored score for quick audit.
    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir,
        filename="jtd-ft-{epoch:02d}-{val_F-measure_beat:.4f}",
        monitor="val_F-measure_beat",
        mode="max",
        save_top_k=args.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )
    # Stop when validation beat F-measure stops improving.
    # This is why runs can end before --max-epochs.
    es_cb = EarlyStopping(
        monitor="val_F-measure_beat",
        mode="max",
        patience=args.patience,
    )
    logger = None
    if args.wandb_project:
        # W&B logging is optional and toggled by --wandb-project.
        # save_dir under output-dir keeps local run artifacts beside checkpoints.
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            save_dir=str(out_dir),
            log_model=True,
        )
        # Capture exact run config for reproducibility and later comparison.
        logger.log_hyperparams(vars(args))

    # Lightning trainer controls hardware, precision, callbacks, and logging.
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        precision=args.precision,
        max_epochs=args.max_epochs,
        # Defensive stability setting for occasional gradient spikes.
        gradient_clip_val=1.0,
        default_root_dir=str(out_dir),
        callbacks=[ckpt_cb, es_cb],
        log_every_n_steps=25,
        logger=logger,
    )

    # Execute train/val loop.
    trainer.fit(model, datamodule=dm)

    # Report checkpoint paths so downstream evaluation can use best checkpoint.
    best_path = ckpt_cb.best_model_path if ckpt_cb.best_model_path else "<none>"
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {ckpt_cb.last_model_path}")

    return 0


if __name__ == "__main__":
    # Speeds up matmul kernels on modern GPUs with acceptable precision tradeoff.
    torch.set_float32_matmul_precision("medium")
    raise SystemExit(main())
