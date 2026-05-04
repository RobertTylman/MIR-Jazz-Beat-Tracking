#!/usr/bin/env python3
"""Build a Beat This!-compatible training dataset from Jazz Trio Database v02.

Creates a processed dataset directory with this structure:

<out-root>/
  annotations/jtd/
    info.json
    single.split
    annotations/beats/<track_id>.beats
  audio/spectrograms/jtd/<track_id>/track.npy

Notes:
- Each .beats file is two columns: beat_time_seconds, beat_value.
- beat_value is 1 for downbeats (metre_auto == 1), otherwise 0.
- Audio files are matched by track_id under --audio-root with supported extensions.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soxr
import torch
from tqdm import tqdm

from beat_this.preprocessing import LogMelSpect, load_audio

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--jtd-root",
        type=Path,
        required=True,
        help="Path to jazz-trio-database-v02 (directories with beats.csv).",
    )
    p.add_argument(
        "--audio-root",
        type=Path,
        required=True,
        help="Root containing audio files named <track_id>.<ext> (flat or recursive).",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output directory for Beat This!-formatted data.",
    )
    p.add_argument(
        "--dataset-name",
        default="jtd",
        help="Dataset subfolder name under annotations/ and spectrograms/ (default: jtd).",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of tracks to assign to val in single.split (default: 0.15).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for train/val split.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute spectrograms and rewrite beats even if they already exist.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Reserved for future parallelism; kept for CLI stability.",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Target sample rate for spectrogram front-end (default: 22050).",
    )
    p.add_argument(
        "--hop-length",
        type=int,
        default=441,
        help="Hop length for spectrogram front-end (default: 441 => 50 fps at 22050 Hz).",
    )
    p.add_argument(
        "--n-mels",
        type=int,
        default=128,
        help="Number of mel bins (default: 128).",
    )
    return p.parse_args()


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


def load_jtd_beats(beats_csv: Path) -> np.ndarray:
    df = pd.read_csv(beats_csv)
    beat_times = df["beats"].to_numpy(dtype=np.float64)
    metre = df["metre_auto"].to_numpy(dtype=np.float64)
    beat_values = (metre == 1.0).astype(np.int64)
    keep = np.isfinite(beat_times)
    beat_times = beat_times[keep]
    beat_values = beat_values[keep]
    order = np.argsort(beat_times)
    beat_times = beat_times[order]
    beat_values = beat_values[order]
    return np.column_stack((beat_times, beat_values))


def write_beats_file(out_beats_path: Path, beats_2col: np.ndarray) -> None:
    out_beats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_beats_path, beats_2col, fmt=["%.9f", "%d"], delimiter="\t")


def compute_spectrogram(
    audio_path: Path,
    spectrogrammer: LogMelSpect,
    target_sr: int,
) -> np.ndarray:
    waveform, sr = load_audio(audio_path)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=-1)
    x_np = np.asarray(waveform, dtype=np.float32)
    if sr != target_sr:
        x_np = soxr.resample(x_np, sr, target_sr, quality="HQ").astype(np.float32)
    x = torch.as_tensor(x_np, dtype=torch.float32)
    with torch.inference_mode():
        spect = spectrogrammer(x).cpu().numpy().astype(np.float32)
    return spect


def main() -> int:
    args = parse_args()

    if not args.jtd_root.is_dir():
        raise SystemExit(f"--jtd-root not found: {args.jtd_root}")
    if not args.audio_root.is_dir():
        raise SystemExit(f"--audio-root not found: {args.audio_root}")
    if not (0.0 < args.val_ratio < 1.0):
        raise SystemExit("--val-ratio must be between 0 and 1")

    dataset = args.dataset_name
    ann_dataset_root = args.out_root / "annotations" / dataset
    beats_out_root = ann_dataset_root / "annotations" / "beats"
    spect_root = args.out_root / "audio" / "spectrograms" / dataset

    beats_out_root.mkdir(parents=True, exist_ok=True)
    spect_root.mkdir(parents=True, exist_ok=True)

    track_dirs = sorted(
        d for d in args.jtd_root.iterdir() if d.is_dir() and (d / "beats.csv").is_file()
    )
    if not track_dirs:
        raise SystemExit(f"No track directories with beats.csv found under {args.jtd_root}")

    spectrogrammer = LogMelSpect(
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        device="cpu",
    )

    processed_ids: list[str] = []
    missing_audio: list[str] = []
    failed: list[tuple[str, str]] = []

    for tdir in tqdm(track_dirs, desc="Building JTD dataset"):
        track_id = tdir.name
        try:
            audio_path = find_audio(track_id, args.audio_root)
            if audio_path is None:
                missing_audio.append(track_id)
                continue

            beats_2col = load_jtd_beats(tdir / "beats.csv")
            out_beats = beats_out_root / f"{track_id}.beats"
            if args.overwrite or not out_beats.exists():
                write_beats_file(out_beats, beats_2col)

            out_spect = spect_root / track_id / "track.npy"
            if args.overwrite or not out_spect.exists():
                out_spect.parent.mkdir(parents=True, exist_ok=True)
                spect = compute_spectrogram(audio_path, spectrogrammer, args.sample_rate)
                np.save(out_spect, spect)

            processed_ids.append(track_id)
        except Exception as e:  # noqa: BLE001
            failed.append((track_id, f"{type(e).__name__}: {e}"))

    if not processed_ids:
        raise SystemExit("No tracks processed successfully. Check audio paths and formats.")

    info_path = ann_dataset_root / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with info_path.open("w") as f:
        json.dump({"has_downbeats": True}, f, indent=2)

    rng = random.Random(args.seed)
    ids = sorted(processed_ids)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * args.val_ratio)))
    val_ids = set(ids[:n_val])

    split_path = ann_dataset_root / "single.split"
    with split_path.open("w") as f:
        for tid in sorted(ids):
            part = "val" if tid in val_ids else "train"
            f.write(f"{tid}\t{part}\n")

    print("\nBuild complete")
    print(f"Processed tracks: {len(processed_ids)}")
    print(f"Missing audio:     {len(missing_audio)}")
    print(f"Failed:            {len(failed)}")
    print(f"Output root:       {args.out_root}")
    print(f"Split file:        {split_path}")

    if missing_audio:
        preview = ", ".join(missing_audio[:10])
        print(f"Missing audio examples: {preview}")
    if failed:
        print("Failures:")
        for tid, msg in failed[:20]:
            print(f"  - {tid}: {msg}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
