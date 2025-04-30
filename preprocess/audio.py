import os
import numpy as np
import soundfile as sf

def load_audio(path):
    wav, sr = sf.read(path)

    # This is here just to check sampling rate, should work even with other rates
    if sr != 16000:
        raise ValueError(f"Sample rate {sr} is not supported. Expected 16000.")

    # Convert to mono if needed
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    return wav, sr

def save_audio(path, wav, sr, clip: float = None):
    # No clipping: save entire file
    if clip is None:
        sf.write(path, wav, sr)
    
    # Clipping: save segments of the file
    else:
        # Calculate number of samples per segment
        clip_samples = int(clip * sr)
        base, ext = os.path.splitext(path)
        total_samples = len(wav)

        # Split into segments
        for idx, start in enumerate(range(0, total_samples, clip_samples)):
            segment = wav[start:start + clip_samples]
            seg_path = f"{base}_seg{idx}{ext}"

            # Throw away last segment if too short
            if len(segment) < clip_samples:
                continue
            sf.write(seg_path, segment, sr)