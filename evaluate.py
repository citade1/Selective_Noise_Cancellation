import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import UNet, AttentionUnet, StackedTransformerEncoder
from data_preparation import SpectrogramDataset
from utils import spectrogram_to_waveform, split_dataset
import argparse

# -----------------
# Evaluation Metric
# -----------------
def snr(pred, target):
    signal_power = torch.mean(target**2)
    noise_power = torch.mean((target - pred)**2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    return snr

@torch.no_grad()
def evaluate(model, dataloader, device, output_dir="eval_outputs", save_audio=False):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    total_snr = 0.0
    total_samples = 0

    for batch_idx, (mixture_spec, clean_spec) in enumerate(tqdm(dataloader, desc="Evaluating")):
        mixture_spec, clean_spec = mixture_spec.to(device), clean_spec.to(device)
        
        est_spec = model(mixture_spec) 

        est_wave = spectrogram_to_waveform(est_spec.squeeze(1)) # shape: (B, T)
        clean_wave = spectrogram_to_waveform(clean_spec.squeeze(1))

        batch_snr = snr(est_wave, clean_wave).item()
        total_snr += batch_snr
        total_samples += 1

        if save_audio:
            for i in range(est_wave.size(0)):
                torchaudio.save(os.path.join(output_dir, f"sample_{batch_idx}_{i}_pred.wav"),
                                est_wave[i].unsqueeze(0).cpu(), 16000)
                torchaudio.save(os.path.join(output_dir, f"sample_{batch_idx}_{i}_clean.wav"),
                                clean_wave[i].unsqueeze(0).cpu(), 16000)
    
    print(f"\n Average SNR over {total_samples} batches: {total_snr/total_samples:.2f} dB")

# Colab override for argparse 
# import sys
# sys.argv = ['evaluate.py', '--model', 'attention','--checkpoint', 'checkpoints/attention_best.pt', '--save_audio']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['unet', 'attention'], default='unet')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='eval_outputs')
    parser.add_argument('--save_audio', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if args.model=="unet":
        model = UNet(in_channels=1, out_channels=1).to(device)
    elif args.model=="attention":
        model = AttentionUnet(
            in_channels=1,
            out_channels=1,
            features=[64,128,256],
            attention_block=StackedTransformerEncoder,
            attn_kwargs={"d_model":256, "n_heads":4, "num_layers":1}
        ).to(device)
    
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Dataset 
    dataset = SpectrogramDataset(
        target_dir="data/targets",
        noise_fsd_dir = "data/noises/fsd50k_clips",
        noise_misc_dir="data/noises/freesound_misc_clips",
        snr_range=(5, 15),
        n_noises=3
    )
    _, _, test_set = split_dataset(dataset, seed=42)
    dataloader = DataLoader(test_set, batch_size=args.batch_size)

    evaluate(model, dataloader, device, output_dir=args.output_dir, save_audio=args.save_audio)