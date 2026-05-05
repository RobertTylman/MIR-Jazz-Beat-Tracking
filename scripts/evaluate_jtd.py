"""
Evaluate Beat This! on the Jazz Trio Database (v02).

For each track directory under --data-root, locate the matching audio file
under --audio-root, run Beat This! inference, score predictions against the
ground-truth beats.csv with mir_eval, and write per-track results to CSV.

The script is resume-safe: rerunning with the same --output skips tracks that
are already present.

Example (NYU HPC):
    python evaluate_jtd.py \
        --data-root  /scratch/$USER/jazz-trio-database-v02 \
        --audio-root /scratch/$USER/jtd-audio \
        --checkpoint ./beat_this/checkpoint/final0.ckpt \
        --output     results/beat_this_jtd.csv \
        --device     cuda
"""

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import mir_eval
import numpy as np
import pandas as pd

from beat_this.inference import File2Beats


AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

# Column name -> exact key returned by mir_eval.beat.evaluate.
# mir_eval spells the continuity scores out in full ("Correct Metric Level
# Continuous"), so a previous version of this script that looked them up by
# their short names ("CMLc", ...) always got NaN back. Always use this map.
BEAT_METRIC_KEYS = {
    "F-measure":        "F-measure",
    "Cemgil":           "Cemgil",
    "Goto":             "Goto",
    "P-score":          "P-score",
    "CMLc":             "Correct Metric Level Continuous",
    "CMLt":             "Correct Metric Level Total",
    "AMLc":             "Any Metric Level Continuous",
    "AMLt":             "Any Metric Level Total",
    "Information gain": "Information gain",
}
BEAT_METRICS = tuple(BEAT_METRIC_KEYS)


def find_audio(track_id: str, audio_root: Path) -> Path | None:
    """Look for <track_id>.<ext> under audio_root (recursive)."""
    for ext in AUDIO_EXTS:
        # fast path: flat layout
        flat = audio_root / f"{track_id}{ext}"
        if flat.is_file():
            return flat
    # fall back to recursive search
    for ext in AUDIO_EXTS:
        hits = list(audio_root.rglob(f"{track_id}{ext}"))
        if hits:
            return hits[0]
    return None


def load_ground_truth(beats_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (gt_beats, gt_downbeats) in seconds.

    beats.csv columns: index, beats, piano, bass, drums, metre_auto
    A row is a downbeat when metre_auto == 1.
    """
    df = pd.read_csv(beats_csv)
    beats = df["beats"].to_numpy(dtype=np.float64)
    mask = df["metre_auto"].to_numpy(dtype=np.float64) == 1.0
    downbeats = beats[mask]
    # mir_eval requires sorted, finite times
    beats = np.sort(beats[np.isfinite(beats)])
    downbeats = np.sort(downbeats[np.isfinite(downbeats)])
    return beats, downbeats


def evaluate_one(gt: np.ndarray, est: np.ndarray,
                 gt_down: np.ndarray | None = None,
                 est_down: np.ndarray | None = None) -> dict:
    """Run mir_eval.beat.evaluate, returning the metrics keyed by short name."""
    if len(gt) == 0 or len(est) == 0:
        return {m: float("nan") for m in BEAT_METRICS}
    scores = mir_eval.beat.evaluate(gt, est,
                                    reference_downbeats=gt_down,
                                    estimated_downbeats=est_down)
    return {col: float(scores.get(key, float("nan")))
            for col, key in BEAT_METRIC_KEYS.items()}


def already_done(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    try:
        prev = pd.read_csv(output_path)
        return set(prev["track_id"].astype(str))
    except Exception:
        return set()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, type=Path,
                   help="Path to jazz-trio-database-v02 (track dirs).")
    p.add_argument("--audio-root", required=True, type=Path,
                   help="Directory containing audio files named <track_id>.<ext>.")
    p.add_argument("--checkpoint", default="final0",
                   help="Beat This! checkpoint path or shortname (default: final0).")
    p.add_argument("--output", required=True, type=Path,
                   help="Per-track results CSV (created/appended).")
    p.add_argument("--device", default="cuda",
                   help="torch device: cuda | cpu | cuda:0 (default: cuda).")
    p.add_argument("--dbn", action="store_true",
                   help="Use madmom DBN postprocessing instead of minimal peak picking.")
    p.add_argument("--float16", action="store_true",
                   help="Use float16 inference (GPU recommended).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only evaluate the first N tracks (for smoke tests).")
    p.add_argument("--track-ids-file", type=Path, default=None,
                   help="Optional newline-delimited track IDs to evaluate (test-only subset).")
    p.add_argument("--save-preds-dir", type=Path, default=None,
                   help="If set, write predicted beats/downbeats per track here.")
    args = p.parse_args()

    if not args.data_root.is_dir():
        sys.exit(f"--data-root not found: {args.data_root}")
    if not args.audio_root.is_dir():
        sys.exit(f"--audio-root not found: {args.audio_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_preds_dir:
        args.save_preds_dir.mkdir(parents=True, exist_ok=True)

    track_dirs = sorted(d for d in args.data_root.iterdir()
                        if d.is_dir() and (d / "beats.csv").is_file())
    if args.track_ids_file:
        if not args.track_ids_file.is_file():
            sys.exit(f"--track-ids-file not found: {args.track_ids_file}")
        wanted = {
            line.strip()
            for line in args.track_ids_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        track_dirs = [d for d in track_dirs if d.name in wanted]
        print(
            f"Applied track-id filter: {len(track_dirs)} tracks from {args.track_ids_file}.",
            flush=True,
        )
    if args.limit:
        track_dirs = track_dirs[:args.limit]
    print(f"Found {len(track_dirs)} track directories.", flush=True)

    done = already_done(args.output)
    if done:
        print(f"Resuming: {len(done)} tracks already in {args.output}.", flush=True)

    fieldnames = ["track_id", "audio_path", "n_gt_beats", "n_pred_beats",
                  "n_gt_downbeats", "n_pred_downbeats", "infer_seconds", "status"]
    fieldnames += [f"beat_{m}" for m in BEAT_METRICS]
    fieldnames += [f"downbeat_{m}" for m in BEAT_METRICS]

    write_header = not args.output.exists()
    out_f = args.output.open("a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        out_f.flush()

    print(f"Loading Beat This! ({args.checkpoint}) on {args.device}...", flush=True)
    file2beats = File2Beats(checkpoint_path=args.checkpoint,
                            device=args.device,
                            float16=args.float16,
                            dbn=args.dbn)
    print("Model ready.", flush=True)

    n_ok = n_skip = n_fail = 0
    for i, tdir in enumerate(track_dirs, 1):
        track_id = tdir.name
        if track_id in done:
            n_skip += 1
            continue

        row = {"track_id": track_id, "audio_path": "",
               "n_gt_beats": 0, "n_pred_beats": 0,
               "n_gt_downbeats": 0, "n_pred_downbeats": 0,
               "infer_seconds": 0.0, "status": ""}
        for m in BEAT_METRICS:
            row[f"beat_{m}"] = float("nan")
            row[f"downbeat_{m}"] = float("nan")

        try:
            audio_path = find_audio(track_id, args.audio_root)
            if audio_path is None:
                row["status"] = "missing_audio"
                n_fail += 1
                writer.writerow(row); out_f.flush()
                print(f"[{i}/{len(track_dirs)}] {track_id}: MISSING AUDIO", flush=True)
                continue
            row["audio_path"] = str(audio_path)

            gt_beats, gt_downbeats = load_ground_truth(tdir / "beats.csv")
            row["n_gt_beats"] = int(len(gt_beats))
            row["n_gt_downbeats"] = int(len(gt_downbeats))

            t0 = time.perf_counter()
            pred_beats, pred_downbeats = file2beats(str(audio_path))
            row["infer_seconds"] = round(time.perf_counter() - t0, 3)
            pred_beats = np.asarray(pred_beats, dtype=np.float64)
            pred_downbeats = np.asarray(pred_downbeats, dtype=np.float64)
            row["n_pred_beats"] = int(len(pred_beats))
            row["n_pred_downbeats"] = int(len(pred_downbeats))

            if args.save_preds_dir:
                np.savez(args.save_preds_dir / f"{track_id}.npz",
                         beats=pred_beats, downbeats=pred_downbeats)

            for m, v in evaluate_one(gt_beats, pred_beats,
                                      gt_downbeats, pred_downbeats).items():
                row[f"beat_{m}"] = v
            for m, v in evaluate_one(gt_downbeats, pred_downbeats).items():
                row[f"downbeat_{m}"] = v

            row["status"] = "ok"
            n_ok += 1
            print(f"[{i}/{len(track_dirs)}] {track_id}: "
                  f"beat_F={row['beat_F-measure']:.3f} "
                  f"db_F={row['downbeat_F-measure']:.3f} "
                  f"({row['infer_seconds']}s)", flush=True)
        except Exception as e:
            n_fail += 1
            row["status"] = f"error: {type(e).__name__}: {e}"
            print(f"[{i}/{len(track_dirs)}] {track_id}: FAILED {row['status']}",
                  flush=True)
            traceback.print_exc()

        writer.writerow(row)
        out_f.flush()

    out_f.close()

    print(f"\nDone. ok={n_ok} skipped={n_skip} failed={n_fail}", flush=True)

    df = pd.read_csv(args.output)
    ok = df[df["status"] == "ok"]
    if len(ok):
        print(f"\nAggregate over {len(ok)} successful tracks:")
        for prefix in ("beat", "downbeat"):
            means = {m: ok[f"{prefix}_{m}"].mean() for m in BEAT_METRICS}
            print(f"  {prefix}: " +
                  ", ".join(f"{m}={means[m]:.3f}" for m in
                            ("F-measure", "CMLt", "AMLt", "Cemgil")))

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
