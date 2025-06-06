import os
import pandas as pd
import torch
from tqdm import tqdm
from utils import load_audio, normalize_audio, to_fixed_length
import argparse

"""
Preprocess FSD50K audio by:
- Filtering relevant labels
- Resampling to 16 kHz
- Normalizing (RMS)
- Segmenting into 5-second chunks
- Saving as .pt tensors
"""


parser = argparse.ArgumentParser()
parser.add_argument("--fsd_audio_dir", type=str, required=True, help="path to FSD50K data")
parser.add_argument("--fsd_meta_path", type=str, required=True, help="path fsd metadata, (filename_label)")
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

# path to FSD50K data. 
FSD_AUDIO_DIR = args.fsd_audio_dir
FSD_META_PATH = args.fsd_meta_path
OUTPUT_DIR = args.output_dir

if not os.path.exists(FSD_AUDIO_DIR):
    raise FileNotFoundError(f"fsd_audio_dir not found: {FSD_AUDIO_DIR}")
if not os.path.exists(FSD_META_PATH):
    raise FileNotFoundError(f"fsd_meta_path not found: {FSD_META_PATH}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# relevant lables in the fsd50k dataset
DESIRED_LABELS = {
    # Vehicles & Subway-related
    "train", "subway_and_metro_and_underground", "rail_transport", "bus", "motor_vehicle_(road)",
    "traffic_noise_and_roadway_noise", "engine", "engine_starting", "idling",

    # Human sounds
    "speech", "conversation", "human_voice",
    "male_speech_and_man_speaking", "female_speech_and_woman_speaking",
    "child_speech_and_kid_speaking", "chatter", "cheering", "laughter", "cough", "sneeze",

    # Crowd and transit ambience
    "crowd", "walk_and_footsteps", "run", "door", "sliding_door",

    # Mechanical ambience
    "brake_squeal", "mechanical_fan", "mechanisms"
}

DEFAULT_SR = 16000
SEGMENT_DURATION = 5.0 # seconds

# load metadata
meta_df = pd.read_csv(FSD_META_PATH)

# Filter rows that contain at least one desired label
def has_desired_label(label_str):
    labels = set(label.strip().lower() for label in label_str.split(','))
    return not DESIRED_LABELS.isdisjoint(labels)

filtered_df = meta_df[meta_df["labels"].apply(has_desired_label)]

print(f"Found {len(filtered_df)} files with desired labels.")

# process and save
for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
    fname = str(row["fname"])
    if not fname.endswith(".wav"):
        fname += ".wav"
    
    path = os.path.join(FSD_AUDIO_DIR, fname)
    if not os.path.isfile(path):
        print(f"[Skip] Missing file: {path}")
        continue


    try:
        waveform, sr = load_audio(path, sr=DEFAULT_SR)
        waveform = normalize_audio(waveform, method="rms", target_level=0.1)

        # Segment and save
        segments = to_fixed_length(waveform, sr, duration_sec=SEGMENT_DURATION)
        for i, segment in enumerate(segments):
            out_path = os.path.join(OUTPUT_DIR, f"{fname}_{i}.pt")
            torch.save(segment, out_path)
    except Exception as e:
        print(f"[Error] Skipped {fname}: {e}")
