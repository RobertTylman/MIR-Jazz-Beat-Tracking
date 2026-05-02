import pandas as pd
from pathlib import Path
import os

def detect_time_signature(beats_csv):
    try:
        df = pd.read_csv(beats_csv)
        if 'metre_auto' not in df.columns:
            return None
        
        # metre_auto == 1 marks downbeats
        downbeat_indices = df[df['metre_auto'] == 1].index.tolist()
        if len(downbeat_indices) < 2:
            return None
        
        # Calculate differences between consecutive downbeat indices
        diffs = [downbeat_indices[i+1] - downbeat_indices[i] for i in range(len(downbeat_indices)-1)]
        if not diffs:
            return None
            
        # Most common difference is likely the beats per bar
        mode_diff = pd.Series(diffs).mode()
        if not mode_diff.empty:
            return int(mode_diff[0])
        return None
    except Exception:
        return None

data_root = Path('jazz-trio-database-v02')
track_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and (d / 'beats.csv').is_file()])

print(f"Scanning {len(track_dirs)} tracks...")
stats = {}
for tdir in track_dirs:
    ts = detect_time_signature(tdir / 'beats.csv')
    if ts:
        stats[ts] = stats.get(ts, 0) + 1

print("\nTime Signature Breakdown (beats per bar):")
for ts, count in sorted(stats.items()):
    print(f"{ts}/4 (or equivalent): {count} tracks")
