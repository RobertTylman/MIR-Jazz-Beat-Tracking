# MIR-Jazz-Beat-Tracking

Improving automatic beat and downbeat tracking for jazz music.

Most state-of-the-art beat trackers are trained on pop, rock, and electronic music, where the pulse is steady and the timing is quantized. Jazz is harder: swing feel, expressive rubato, brushed drums, walking bass, and frequent tempo modulation all push these models out of distribution. This project measures how badly existing models break on jazz, and then asks whether we can fix it by training on jazz data directly.

## Aim

1. **Benchmark** the existing [madmom](https://github.com/CPJKU/madmom) beat tracking model on the [Jazz Trio Database](https://zenodo.org/records/13828030) to establish a baseline for how well a general-purpose tracker handles jazz.
2. **Retrain / fine-tune** the model on the Jazz Trio Database.
3. **Re-evaluate** the retrained model against the same baseline to quantify the improvement.

For comparison, we also evaluate other widely used beat trackers — `librosa` and [Beat This!](https://github.com/CPJKU/beat_this) — under the same protocol.

## Dataset

The **Jazz Trio Database** (Cheston et al., 2024) is a corpus of jazz piano-trio recordings (piano, bass, drums) with frame-level annotations for beats, downbeats, and per-instrument onsets.

### Version 02 (Expanded Local Corpus)
This project utilizes the `jazz-trio-database-v02`, which contains **1294 jazz trio performances**. It spans recordings from the 1950s–2010s, providing meaningful coverage of jazz performance practice — swing eighths, tempo drift, soloist push/pull against the rhythm section, and recording-era timbral differences.

Each performance directory within the database contains:
- `beats.csv`: Human-verified beat and downbeat annotations.
- `piano_onsets.csv`, `bass_onsets.csv`, `drums_onsets.csv`: Onset times for each instrument.
- `piano_midi.mid`: MIDI representation of the piano performance.
- `metadata.json`: Track details including artist, album, year, and estimated tempo.

- Zenodo Record (v1): <https://zenodo.org/records/13828030>
- Beat This! pretrained model checkpoint: <https://cloud.cp.jku.at/index.php/s/7ik4RrBKTS273gp>

## Repository contents

- `evaluation/` — model-specific analysis notebooks and evaluation CSVs.
  - `graphs-beatthis.ipynb`: Visualization and metric analysis for Beat This!.
  - `graphs-madmom.ipynb`: Visualization and metric analysis for madmom.
  - `inference_test.ipynb`: Interactive sandbox for testing model inference.
  - `csvs/`: Raw evaluation metrics in CSV format.
- `scripts/` — support scripts for evaluation and inference.
  - `run_beat_this.py`: Wrapper for Beat This! inference.
  - `evaluate_jtd.py`: Full dataset evaluation for Beat This!.
  - `evaluate_madmom.py`: Full dataset evaluation for madmom.
  - `support_function.py`: Plotting utilities.
- `beat_this/` — core model implementation and checkpoints.
- `jazz-trio-database-v02/` — dataset annotations.
- `test_audio/` — sample audio clips for inference testing.

## Advisor

Brian McFee (NYU Music and Audio Research Lab).
