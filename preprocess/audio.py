import soundfile as sf
import numpy as np

def load_audio(path):
    wav, sr = sf.read(path)
    # Convert to mono if needed
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    return wav, sr

def save_audio(path, wav, sr):
    sf.write(path, wav, sr)