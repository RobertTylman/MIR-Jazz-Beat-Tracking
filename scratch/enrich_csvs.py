import pandas as pd
from pathlib import Path

def get_time_signature(track_id, data_root):
    beats_csv = Path(data_root) / track_id / "beats.csv"
    if not beats_csv.is_file():
        return 4
    try:
        df = pd.read_csv(beats_csv)
        if 'metre_auto' not in df.columns:
            return 4
        downbeat_indices = df[df['metre_auto'] == 1].index.tolist()
        if len(downbeat_indices) < 2:
            return 4
        diffs = [downbeat_indices[i+1] - downbeat_indices[i] for i in range(len(downbeat_indices)-1)]
        if not diffs:
            return 4
        return int(pd.Series(diffs).mode()[0])
    except:
        return 4

def enrich_csv(csv_path, data_root):
    df = pd.read_csv(csv_path)
    print(f"Enriching {csv_path}...")
    df['time_signature'] = df['track_id'].apply(lambda x: get_time_signature(x, data_root))
    df.to_csv(csv_path, index=False)
    print("Done.")

data_root = "jazz-trio-database-v02"
enrich_csv("evaluation/csvs/madmom_jtd.csv", data_root)
enrich_csv("evaluation/csvs/beat_this_jtd.csv", data_root)
