import os
import glob
import tqdm
import random
import argparse
import librosa
import tarfile
import zipfile
import urllib.request
import numpy as np

from audio import load_audio, save_audio

MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
RIRS_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"

# Set random seed for reproducibility
random.seed(42)

############################
# MUSAN and RIR - not used #
############################

def download_and_extract_musan(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    archive = os.path.join(target_dir, 'musan.tar.gz')
    print(f"Downloading MUSAN to {archive}...")
    urllib.request.urlretrieve(MUSAN_URL, archive)
    print("Extracting noise files...")
    with tarfile.open(archive, 'r:gz') as tar:
        def wav_filter(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
            if member.isfile() and member.name.endswith('.wav') and member.name.startswith('musan/noise/'):
                return member
            return None
        tar.extractall(path=target_dir, filter=wav_filter)
    os.remove(archive)
    print(f"MUSAN noise ready: {target_dir}/musan/noise")

def download_and_extract_rirs(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    archive = os.path.join(target_dir, 'rirs_noises.zip')
    print(f"Downloading RIRS_NOISES to {archive}...")
    urllib.request.urlretrieve(RIRS_URL, archive)
    print("Extracting RIR files...")
    with zipfile.ZipFile(archive, 'r') as z:
        for member in z.namelist():
            if member.startswith('RIRS_NOISES/simulated_rirs/') and member.endswith('.wav'):
                z.extract(member, path=target_dir)
    os.remove(archive)
    print(f"RIR data ready: {target_dir}/RIRS_NOISES/simulated_rirs")

def ensure_noise_data(download_dir):
    if not os.path.exists(os.path.join(download_dir, 'musan')):
        download_and_extract_musan(download_dir)


def ensure_rir_data(download_dir):
    if not os.path.exists(os.path.join(download_dir, 'RIRS_NOISES')):
        download_and_extract_rirs(download_dir)

def add_noise(wav, noise_wav, snr):
    # Scale noise to achieve target SNR
    sig_pow = np.mean(wav**2)
    noise_pow = np.mean(noise_wav**2)
    noise_wav = noise_wav * np.sqrt(sig_pow / (10**(snr/10) * noise_pow))
    # Repeat or trim noise to match length
    if len(noise_wav) < len(wav):
        noise_wav = np.pad(noise_wav, (0, len(wav)-len(noise_wav)), mode='wrap')
    else:
        noise_wav = noise_wav[:len(wav)]
    return wav + noise_wav

def add_reverb(wav, rir):
    # Convolve and trim to original length
    rv = np.convolve(wav, rir, mode='full')
    return rv[:len(wav)]

############################

def augment_file(filepath, out_dir):
    wav, sr = load_audio(filepath)
    base = os.path.splitext(os.path.basename(filepath))[0]

    # 1. Original
    save_audio(os.path.join(out_dir, f"{base}.wav"), wav, sr)

    # 2. Speed perturbation
    for rate in [0.9, 1.1]:
        wav_sp = librosa.effects.time_stretch(wav, rate=rate)
        save_audio(os.path.join(out_dir, f"{base}_sp{rate:.1f}.wav"), wav_sp, sr)

    # 3. Pitch shift
    for steps in [-1, 1]:
        wav_ps = librosa.effects.pitch_shift(wav, sr=sr, n_steps=steps)
        save_audio(os.path.join(out_dir, f"{base}_ps{steps:+d}.wav"), wav_ps, sr)

    # These steps were skipped due to need for external data

    # # 4. Additive noise
    # if noise_files:
    #     for snr in [5, 10, 15]:
    #         noise, _ = load_audio(random.choice(noise_files))
    #         wav_no = add_noise(wav, noise, snr)
    #         save_audio(os.path.join(out_dir, f"{base}_no{snr}dB.wav"), wav_no, sr)
    # else:
    #     pass

    # # 5. Reverberation
    # if rir_files:
    #     rir_f = random.choice(rir_files)
    #     rir, _ = load_audio(rir_f)
    #     wav_rv = add_reverb(wav, rir)
    #     save_audio(os.path.join(out_dir, f"{base}_rv.wav"), wav_rv, sr)
    # else:
    #     pass

def process_folder(input_dir, output_dir, download_dir):
    # # Ensure noise and RIR data are available
    # ensure_noise_data(download_dir)
    # ensure_rir_data(download_dir)

    # global noise_files
    # noise_dir = os.path.join(download_dir, "musan", "noise")
    # noise_files = glob.glob(os.path.join(noise_dir, "free-sound", "*.wav")) + \
    #     glob.glob(os.path.join(noise_dir, "sound-bible", "*.wav"))

    # global rir_files
    # rir_dir = os.path.join(download_dir, "RIRS_NOISES", "simulated_rirs")
    # rir_files = []
    # for folder in ["largeroom", "mediumroom", "smallroom"]:
    #     subfolder = os.path.join(rir_dir, folder)
    #     for subsubfolder in os.listdir(subfolder):
    #         rir_files += glob.glob(os.path.join(subfolder, subsubfolder, "*.wav"))
    
    # print(f"Found {len(noise_files)} noise files and {len(rir_files)} RIR files.")

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

            for wav_path in glob.glob(os.path.join(spk_in, "*.wav")):
                augment_file(wav_path, spk_out)

def main():
    parser = argparse.ArgumentParser(description="Data augmentation")
    parser.add_argument("-i", "--input",    required=True,  help="Path to original data directory")
    parser.add_argument("-o", "--output",   required=True,  help="Path to augmented data directory")
    parser.add_argument("-d", "--download", default="data", help="Path to download directory for noise and RIR data (not used)")
    args = parser.parse_args()

    process_folder(args.input, args.output, args.download)

if __name__ == '__main__':
    main()
