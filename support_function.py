import librosa
import numpy as np
import matplotlib.pyplot as plt


def plot_beats(audio_path, beats, downbeats):
    """
    Plot audio waveform with beat and downbeat markers.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.
    beats : np.ndarray
        Beat timestamps in seconds.
    downbeats : np.ndarray
        Downbeat timestamps in seconds.
    """
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    time = np.linspace(0, len(audio) / sr, num=len(audio))

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(time, audio, color='grey', linewidth=0.5, label='Waveform')
    ax.vlines(beats, ymin=audio.min(), ymax=audio.max(),
              colors='steelblue', linewidths=0.8, alpha=0.7, label='Beats')
    ax.vlines(downbeats, ymin=audio.min(), ymax=audio.max(),
              colors='red', linewidths=1.2, alpha=0.9, label='Downbeats')

    ax.set_title('Audio Waveform with Beat Tracking')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend(frameon=True, framealpha=1.0, edgecolor='black')

    plt.tight_layout()
    plt.show()


