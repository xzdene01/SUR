import os
import numpy as np
import soundfile as sf

def load_audio(path):
    wav, sr = sf.read(path)

    # Convert to mono if needed
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    return wav, sr

def save_audio(path, wav, sr, clip: float = None):
    if clip is None:
        sf.write(path, wav, sr)
    else:
        clip_samples = int(clip * sr)
        base, ext = os.path.splitext(path)
        total_samples = len(wav)

        for idx, start in enumerate(range(0, total_samples, clip_samples)):
            segment = wav[start:start + clip_samples]
            seg_path = f"{base}_seg{idx}{ext}"

            # Throw away last segment if too short
            if len(segment) < clip_samples:
                continue

            sf.write(seg_path, segment, sr)