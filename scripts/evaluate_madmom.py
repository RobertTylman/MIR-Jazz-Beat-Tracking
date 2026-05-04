"""
Evaluate madmom's DBN beat tracker on the Jazz Trio Database.
"""

# ==============================================================================
# MONKEY PATCHES FOR MODERN PYTHON (3.10+) AND NUMPY (1.24+)
# ==============================================================================
import collections
import collections.abc
import numpy as np

collections.MutableSequence = collections.abc.MutableSequence
np.float = float
np.int = int

# This fixes the "setting an array element with a sequence" error when 
# madmom tries to test multiple time signatures (like 3/4 and 4/4).
_original_asarray = np.asarray
def _patched_asarray(a, dtype=None, order=None, *, device=None, copy=None, like=None):
    try:
        return _original_asarray(a, dtype=dtype, order=order, like=like)
    except ValueError as e:
        if "inhomogeneous shape" in str(e) or "sequence" in str(e):
            return _original_asarray(a, dtype=object, order=order, like=like)
        raise e
np.asarray = _patched_asarray
# ==============================================================================

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import mir_eval
import pandas as pd
from madmom.processors import SequentialProcessor
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

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
    for ext in AUDIO_EXTS:
        flat = audio_root / f"{track_id}{ext}"
        if flat.is_file():
            return flat
    for ext in AUDIO_EXTS:
        hits = list(audio_root.rglob(f"{track_id}{ext}"))
        if hits:
            return hits[0]
    return None


def load_ground_truth(beats_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(beats_csv)
    beats = df["beats"].to_numpy(dtype=np.float64)
    mask = df["metre_auto"].to_numpy(dtype=np.float64) == 1.0
    downbeats = beats[mask]
    beats = np.sort(beats[np.isfinite(beats)])
    downbeats = np.sort(downbeats[np.isfinite(downbeats)])
    return beats, downbeats


def evaluate_one(gt: np.ndarray, est: np.ndarray,
                 gt_down: np.ndarray | None = None,
                 est_down: np.ndarray | None = None) -> dict:
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--audio-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--downbeats", action="store_true",
                   help="Track downbeats instead of beats.")
    p.add_argument("--fps", type=int, default=100,
                   help="Processing FPS (default: 100).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--include", nargs="+", help="Only evaluate these track IDs.")
    p.add_argument("--exclude", nargs="+", help="Skip these track IDs.")
    p.add_argument("--include-from", type=Path, help="File containing track IDs to evaluate (one per line).")
    p.add_argument("--exclude-from", type=Path, help="File containing track IDs to skip (one per line).")
    p.add_argument("--match", help="Only evaluate track IDs containing this string.")
    p.add_argument("--force", action="store_true", help="Re-evaluate even if track is already in output.")
    args = p.parse_args()

    if not args.data_root.is_dir():
        sys.exit(f"--data-root not found: {args.data_root}")
    if not args.audio_root.is_dir():
        sys.exit(f"--audio-root not found: {args.audio_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    track_dirs = sorted(d for d in args.data_root.iterdir()
                        if d.is_dir() and (d / "beats.csv").is_file())
    
    # Apply include/exclude filters
    if args.include:
        track_dirs = [d for d in track_dirs if d.name in args.include]
    if args.include_from:
        with args.include_from.open() as f:
            include_set = {line.strip() for line in f if line.strip()}
        track_dirs = [d for d in track_dirs if d.name in include_set]
    
    if args.exclude:
        track_dirs = [d for d in track_dirs if d.name not in args.exclude]
    if args.exclude_from:
        with args.exclude_from.open() as f:
            exclude_set = {line.strip() for line in f if line.strip()}
        track_dirs = [d for d in track_dirs if d.name not in exclude_set]

    if args.match:
        track_dirs = [d for d in track_dirs if args.match in d.name]

    if args.limit:
        track_dirs = track_dirs[:args.limit]
    print(f"Found {len(track_dirs)} track directories to process.", flush=True)

    done = already_done(args.output)
    if done and not args.force:
        print(f"Resuming: {len(done)} tracks already in {args.output}. Use --force to re-evaluate.", flush=True)


    # Initialize madmom processor
    if args.downbeats:
        rnn = RNNDownBeatProcessor()
        # Enable 3/4 and 4/4 time signatures!
        dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=args.fps)
        tracker = SequentialProcessor([rnn, dbn])
    else:
        rnn = RNNBeatProcessor()
        dbn = DBNBeatTrackingProcessor(fps=args.fps)
        tracker = SequentialProcessor([rnn, dbn])
    
    print(f"Loaded madmom {'downbeat' if args.downbeats else 'beat'} tracker.", flush=True)

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

    n_ok = n_skip = n_fail = 0
    for i, tdir in enumerate(track_dirs, 1):
        track_id = tdir.name
        if track_id in done and not args.force:
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
            # Convert Path to string for madmom!
            activations = tracker(str(audio_path))
            row["infer_seconds"] = round(time.perf_counter() - t0, 3)

            if args.downbeats:
                # DBNDownBeatTrackingProcessor returns (time, beat_number)
                # Beat numbers are 1, 2, 3, 4. Downbeats are where beat_number == 1.
                pred_beats = activations[:, 0]
                pred_downbeats = activations[activations[:, 1] == 1, 0]
            else:
                # DBNBeatTrackingProcessor returns just a 1D array of times
                pred_beats = activations
                pred_downbeats = np.array([])

            pred_beats = np.asarray(pred_beats, dtype=np.float64)
            pred_downbeats = np.asarray(pred_downbeats, dtype=np.float64)
            row["n_pred_beats"] = int(len(pred_beats))
            row["n_pred_downbeats"] = int(len(pred_downbeats))

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
