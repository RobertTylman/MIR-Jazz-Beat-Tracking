#!/usr/bin/env python3
"""Create a strict train/val/test split for JTD and write split artifacts.

Outputs:
1) Beat This split file at:
   <data-dir>/annotations/<dataset-name>/single.split
   containing only `train` and `val` rows (test rows are excluded on purpose).
2) Track ID lists under --out-dir:
   - train_ids.txt
   - val_ids.txt
   - test_ids.txt

Use `test_ids.txt` with scripts/evaluate_jtd.py --track-ids-file for unbiased test evaluation.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--jtd-root",
        type=Path,
        required=True,
        help="Path to jazz-trio-database-v02 (track dirs with beats.csv).",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Processed Beat This dataset root (e.g., /scratch/$USER/beatthis_jtd).",
    )
    p.add_argument(
        "--dataset-name",
        default="jtd",
        help="Dataset folder under annotations/ (default: jtd).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for split id files (train_ids.txt, val_ids.txt, test_ids.txt).",
    )
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-9:
        raise SystemExit(f"Ratios must sum to 1.0, got {total}")
    if not args.jtd_root.is_dir():
        raise SystemExit(f"--jtd-root not found: {args.jtd_root}")
    if not args.data_dir.is_dir():
        raise SystemExit(f"--data-dir not found: {args.data_dir}")

    track_ids = sorted(
        d.name for d in args.jtd_root.iterdir() if d.is_dir() and (d / "beats.csv").is_file()
    )
    if not track_ids:
        raise SystemExit(f"No tracks found under {args.jtd_root}")

    rng = random.Random(args.seed)
    ids = track_ids[:]
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * args.train_ratio))
    n_val = int(round(n * args.val_ratio))
    # Ensure no loss due to rounding.
    n_test = n - n_train - n_val

    train_ids = sorted(ids[:n_train])
    val_ids = sorted(ids[n_train : n_train + n_val])
    test_ids = sorted(ids[n_train + n_val :])
    assert len(train_ids) + len(val_ids) + len(test_ids) == n

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_ids.txt").write_text("\n".join(train_ids) + "\n")
    (args.out_dir / "val_ids.txt").write_text("\n".join(val_ids) + "\n")
    (args.out_dir / "test_ids.txt").write_text("\n".join(test_ids) + "\n")

    split_path = args.data_dir / "annotations" / args.dataset_name / "single.split"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w") as f:
        for tid in train_ids:
            f.write(f"{tid}\ttrain\n")
        for tid in val_ids:
            f.write(f"{tid}\tval\n")

    print("Split created")
    print(f"total={n} train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    print(f"single.split: {split_path}")
    print(f"ids dir: {args.out_dir}")
    print("Use test_ids.txt with evaluate_jtd.py --track-ids-file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

