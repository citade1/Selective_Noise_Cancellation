import os
import torch
from utils import load_audio, normalize_audio, mp3_to_wav, to_fixed_length

# path to subway announcment data
RAW_DIR = "/Volumes/Samsung_T5/data_snc/subway_announcement"
CLEAN_WAV_DIR = "/Volumes/Samsung_T5/data_snc/targets"

os.makedirs(CLEAN_WAV_DIR, exist_ok=True)

DEFAULT_SR = 16000
SEGMENT_DURATION = 5.0 # seconds

# Traverse all line folders and process mp3s
for line_folder in os.listdir(RAW_DIR): # line 1, line 2, ...
    line_path = os.path.join(RAW_DIR, line_folder) # e.g. data/subway_announcement/line1
    if not os.path.isdir(line_path):
        continue
    
    for fname in os.listdir(line_path):
        if fname.startswith("._") or not fname.endswith(".mp3"):
            continue
        
        mp3_path = os.path.join(line_path, fname) 
        
        # change mp3 file to wav
        wav_filename = fname.replace(".mp3", ".wav")
        wav_path = os.path.join(CLEAN_WAV_DIR, wav_filename)

        try:
            # convert and load 
            mp3_to_wav(mp3_path, wav_path)
            waveform, sr = load_audio(wav_path, DEFAULT_SR)
            waveform = normalize_audio(waveform, method="rms", target_level=0.1)
            os.remove(wav_path) # remove temp .wav file

            # Segment and save
            segments = to_fixed_length(waveform, sr, duration_sec=SEGMENT_DURATION)
            base_name = fname.replace(".mp3", "")
            for i, segment in enumerate(segments):
                out_path = os.path.join(CLEAN_WAV_DIR, f"{base_name}_{i}.pt")
                torch.save(segment, out_path)
            
        except Exception as e:
            print(f"[Error] Skipped {mp3_path}: {e}")
