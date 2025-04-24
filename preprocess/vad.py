import os
import argparse
import glob
from pydub import AudioSegment, silence
import tqdm

from audio import load_audio, save_audio

#################
# Configuration #
#################

MIN_SILENCE_LEN = 200   # Minimum length of silence to consider it as a pause (ms)
SILENCE_THRESH  = -40   # Amplitude threshold below which is silence (dBFS)
PAD_MS          = 200   # Padding around detected segments (ms)
MERGE_MS        = 250   # Maximum gap to merge segments (ms)
MIN_SEG_LEN_MS  = 1000  # Minimum length of segments to keep, otherwise merge (ms)


def enforce_min_length(segments):
    final = []
    for idx, (s, e) in enumerate(segments):
        length = e - s
        # This segment is long enough
        if length >= MIN_SEG_LEN_MS:
            final.append([s, e])
        
        else:
            # Merge with previous
            if final:
                final[-1][1] = e
            
            else:
                # No previous -> merge into next if exists
                if idx + 1 < len(segments):
                    segments[idx+1][0] = s
                
                # Singel short segment
                else:
                    final.append([s, e])
    return final

def process_file(in_path, out_dir):
    audio = AudioSegment.from_wav(in_path)
    # Detect non-silent intervals [(start_ms, end_ms), ...]
    intervals = silence.detect_nonsilent(
        audio,
        min_silence_len=MIN_SILENCE_LEN,
        silence_thresh=SILENCE_THRESH
    )

    # No non-silent intervals found
    if not intervals:
        return

    # Pad each interval
    duration = len(audio)
    padded = [(max(0, s - PAD_MS), min(duration, e + PAD_MS)) for s, e in intervals]

    # Merge segments that are close together (within MERGE_MS)
    merged = []
    for seg in padded:
        # Append the first segment
        if not merged:
            merged.append(list(seg))
        
        else:
            _, prev_end = merged[-1]
            curr_start, curr_end = seg
            if curr_start - prev_end <= MERGE_MS:
                merged[-1][1] = curr_end
            else:
                merged.append([curr_start, curr_end])

    merged = enforce_min_length(merged)

    basename = os.path.splitext(os.path.basename(in_path))[0]
    for idx, (s, e) in enumerate(merged):
        segment = audio[s:e]
        out_path = os.path.join(out_dir, f"{basename}_seg{idx}.wav")
        segment.export(out_path, format="wav")

def process_folder(input_dir, output_dir):
    for subset in ['train', 'dev']:
        in_dir = os.path.join(input_dir, subset)
        if not os.path.isdir(in_dir):
            print(f"Input directory {in_dir} does not exist.")
            continue

        for speaker in tqdm.tqdm(os.listdir(in_dir), desc=f"Processing {subset}"):
            spk_in = os.path.join(in_dir, speaker)
            if not os.path.isdir(spk_in):
                print(f"Speaker directory {spk_in} does not exist.")
                continue

            spk_out = os.path.join(output_dir, subset, speaker)
            os.makedirs(spk_out, exist_ok=True)

            for wav_path in glob.glob(os.path.join(spk_in, '*.wav')):
                process_file(wav_path, spk_out)

def main():
    parser = argparse.ArgumentParser(description="PyDub-based VAD")
    parser.add_argument('-i', '--input', required=True, help="Input root folder (augmented data)")
    parser.add_argument('-o', '--output', required=True, help="Output root folder (speech-only)")
    args = parser.parse_args()

    process_folder(args.input, args.output)

if __name__ == '__main__':
    main()
