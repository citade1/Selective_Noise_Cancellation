import os
import torch
from utils import load_audio, normalize_audio, to_fixed_length

# Path
INPUT_DIR = "/Volumes/Samsung_T5/data_snc/freesound_misc_clips"
OUTPUT_DIR = "/Volumes/Samsung_T5/data_snc/noises/fsd_misc_clips"

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEFAULT_SR = 16000
SEGMENT_DURATION = 5.0 # seconds

for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if file.startswith("._") or not file.endswith((".wav", ".mp3", ".flac")):
            continue
        try:
            # load 
            file_path = os.path.join(root, file)
            waveform, sr = load_audio(file_path, DEFAULT_SR)
            waveform = normalize_audio(waveform, method="rms", target_level=0.1)
            
            # Segment and save
            segments = to_fixed_length(waveform, sr, duration_sec=SEGMENT_DURATION)
            base_name = os.path.splitext(file)[0]
            for i, segment in enumerate(segments):
                out_path = os.path.join(OUTPUT_DIR, f"{base_name}_{i}.pt")
                torch.save(segment, out_path)

        except Exception as e:
            print(f"[Error] skipped {file_path} : {e}")
