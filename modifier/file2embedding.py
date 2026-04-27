import torch
import soxr
from beat_this.inference import load_model, split_piece
from beat_this.preprocessing import LogMelSpect, load_audio

class File2Embeddings:
    """
    Runs Beat_This inference and returns the transformer embedding for an audio file.

    """

    CHUNK_SIZE = 1500
    BORDER_SIZE = 6

    def __init__(self, checkpoint_path, device="cpu", dbn=False):
        self.device = torch.device(device)
        self.model = load_model(checkpoint_path, self.device)
        self.spect_fn = LogMelSpect(device=self.device)

    def __call__(self, audio_path):
        signal, sr = load_audio(audio_path)

        #incase of stereo, convert to mono
        if signal.ndim == 2:
            signal = signal.mean(1)   # stereo → mono
        if sr != 22050:
            signal = soxr.resample(signal, in_rate=sr, out_rate=22050)
        signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        spect = self.spect_fn(signal)  # (time, 128)

        # hook captures transformer_blocks output: (batch, time, 512)
        # capture embedding
        captured = {}

        #hook function to capture the output of transformer, 512-dim embedding per frame
        def hook_fn(module, inp, out):
            captured['emb'] = out.detach()
        
        hook = self.model.transformer_blocks.register_forward_hook(hook_fn)
        full_size = spect.shape[0]
        embedding = torch.zeros((full_size, 512), device=self.device)

        chunks, starts = split_piece(
            spect, chunk_size=self.CHUNK_SIZE,
            border_size=self.BORDER_SIZE, avoid_short_end=True,
        )
        b = self.BORDER_SIZE

        with torch.inference_mode():
            for start, chunk in zip(starts, chunks):
                pred = self.model(chunk.unsqueeze(0))
                emb_c = captured['emb'][0][b:-b]      # (trimmed_time, 512)

                s = start + b
                e = start + self.CHUNK_SIZE - b
                embedding[s:e] = emb_c

        hook.remove()
        return embedding.cpu()
