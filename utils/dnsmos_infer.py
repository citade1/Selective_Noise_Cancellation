import numpy as np
import onnxruntime as ort
import soundfile as sf
import os

DEFAULT_SR = 16000
TARGET_LEN = 144160

# Set model path 
MODEL_PATH = os.path.join("models", "dnsmos", "sig_bak_ovr.onnx")

# Load onnx runtime sessions
sess = ort.InferenceSession(MODEL_PATH)

def run_dnsmos(wav_path):
    """
    Run DNSMOS inference using sig_bak_ovr.onnx.
    Returns dict with SIG, BAK, and ovr scores.
    """
    audio, sr = sf.read(wav_path) # SoundFile loads audio as [time, channels]

    if sr != DEFAULT_SR:
         raise ValueError(f"Expected 16kHz audio, got {sr} Hz")

    # Ensure mono (e.g. (80000,))
    if len(audio.shape) > 1:
         audio = np.mean(audio, axis=1)
    
    # Trim or pad audio to 144160 samples (DNSMOS expected length)
    if len(audio) > TARGET_LEN:
         audio = audio[: TARGET_LEN]
    else:
         audio = np.pad(audio, (0, TARGET_LEN - len(audio)), mode="constant")

    audio = np.expand_dims(audio.astype(np.float32), axis=0) # shape: (1, 144000)

    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: audio})[0]

    sig, bak, ovr = outputs[0]

    return {"SIG": float(sig), "BAK": float(bak), "OVR": float(ovr)}