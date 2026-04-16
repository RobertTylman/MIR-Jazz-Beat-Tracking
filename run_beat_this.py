from beat_this.inference import File2Beats


def run_beat_this(checkpoint_path, audio_path):
    #run beat_this with checkpoint
    file2beats = File2Beats(checkpoint_path=checkpoint_path, device="cpu", dbn=False)
    beats, downbeats = file2beats(audio_path)

    return beats, downbeats


