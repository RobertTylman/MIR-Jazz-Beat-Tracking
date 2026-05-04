#!/usr/bin/env python3
"""Fine-tune Beat This! on a BeatDataModule-formatted JTD dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from beat_this.dataset.dataset import BeatDataModule
from beat_this.model.pl_module import PLBeatThis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True, help="Processed dataset root.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="beat_this/checkpoint/final0.ckpt",
        help="Pretrained Beat This! checkpoint to initialize from.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/jtd_finetune"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--train-length", type=int, default=1500)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-epochs", type=int, default=40)
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
    return p.parse_args()


def parse_devices(raw: str):
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
    pl.seed_everything(args.seed, workers=True)

    if not args.data_dir.is_dir():
        raise SystemExit(f"--data-dir not found: {args.data_dir}")

    # Keep BeatDataModule from skipping the only dataset during fit.
    # test_dataset is excluded from train/val split construction.
    dm = BeatDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_length=args.train_length,
        spect_fps=50,
        test_dataset="__none__",
    )

    dm.setup("fit")
    if not dm.train_items:
        raise SystemExit(
            "No training items found. Check annotations/<dataset>/single.split and data-dir layout."
        )

    pos_weights = dm.get_train_positive_weights()

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

    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir,
        filename="jtd-ft-{epoch:02d}-{val_F-measure_beat:.4f}",
        monitor="val_F-measure_beat",
        mode="max",
        save_top_k=args.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )
    es_cb = EarlyStopping(
        monitor="val_F-measure_beat",
        mode="max",
        patience=args.patience,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        precision=args.precision,
        max_epochs=args.max_epochs,
        gradient_clip_val=1.0,
        default_root_dir=str(out_dir),
        callbacks=[ckpt_cb, es_cb],
        log_every_n_steps=25,
    )

    trainer.fit(model, datamodule=dm)

    best_path = ckpt_cb.best_model_path if ckpt_cb.best_model_path else "<none>"
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {ckpt_cb.last_model_path}")

    return 0


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    raise SystemExit(main())
