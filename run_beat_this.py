from beat_this.inference import File2Beats
from support_function import plot_beats, File2Embeddings


def run_beat_this(checkpoint_path, audio_path):
    file2beats = File2Beats(checkpoint_path=checkpoint_path, device="cpu", dbn=False)
    beats, downbeats = file2beats(audio_path)
    return beats, downbeats

