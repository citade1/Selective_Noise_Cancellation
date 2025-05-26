import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import UNet, AttentionUNet, StackedTransformerEncoder
from data_preparation import SpectrogramDataset
from utils import spectrogram_to_waveform, split_dataset
import argparse

# -----------------
# Evaluation Metric
# -----------------
def snr(pred, target):
    signal_power = torch.mean(target**2)
    noise_power = torch.mean((target - pred)**2)
    if noise_power < 1e-10:
        return torch.tensor(float("inf")) # perfect match 
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    return snr

@torch.no_grad()
def evaluate(model, dataloader, device, output_dir="eval_outputs", save_audio=False, save_every=10, snr_thresh=None):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    total_snr = 0.0
    total_samples = 0

    for batch_idx, (mixture_spec, clean_spec) in enumerate(tqdm(dataloader, desc="Evaluating")):
        mixture_spec, clean_spec = mixture_spec.to(device), clean_spec.to(device)
        
        est_spec = model(mixture_spec) 

        est_wave = spectrogram_to_waveform(est_spec.squeeze(1)) # shape: (B, T)
        clean_wave = spectrogram_to_waveform(clean_spec.squeeze(1))

        
        for i in range(est_wave.size(0)):
            snr_value = snr(est_wave, clean_wave).item()
            total_snr += snr_value
            total_samples += 1
            
            if save_audio:
                should_save = ((total_samples % save_every==0) or
                               (snr_thresh is not None and snr_value >= snr_thresh))
                if should_save:
                    torchaudio.save(os.path.join(output_dir, f"sample_{batch_idx}_{i}_pred.wav"),
                                    est_wave[i].unsqueeze(0).cpu(), 16000)
                    torchaudio.save(os.path.join(output_dir, f"sample_{batch_idx}_{i}_clean.wav"),
                                    clean_wave[i].unsqueeze(0).cpu(), 16000)
    avg_snr = total_snr / total_samples if total_samples > 0 else float('nan')
    print(f"\n Average SNR over {total_samples} samples: {avg_snr:.2f} dB")

# -----------
# Entry Point
# -----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['unet', 'attention'], default='unet')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='eval_outputs')
    parser.add_argument('--save_audio', action='store_true')
    parser.add_argument('--save_every', type=int, default=10, help="Save every Nth sample")
    parser.add_argument('--snr_thresh', type=float, default=None, help="Optional SNR threshold to save good samples")
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--noise_fsd_dir', type=str, required=True)
    parser.add_argument('--noise_misc_dir', type=str, required=True)
    parser.add_argument('--snr_min', type=float, default=-5.0)
    parser.add_argument('--snr_max', type=float, default=5.0)
    parser.add_argument('--n_noises', type=int, default=3)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load model
    if args.model=="unet":
        model = UNet(in_channels=1, out_channels=1).to(device)
    elif args.model=="attention":
        model = AttentionUNet(
            in_channels=1,
            out_channels=1,
            features=[64,128,256],
            attention_block=StackedTransformerEncoder,
            attn_kwargs={"d_model":256, "n_heads":4, "dim_ff":512, "num_layers":1}
        ).to(device)
    
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Dataset 
    dataset = SpectrogramDataset(
        target_dir=args.target_dir,
        noise_fsd_dir = args.noise_fsd_dir,
        noise_misc_dir=args.noise_misc_dir,
        snr_range=(args.snr_min, args.snr_max),
        n_noises=args.n_noises
    )
    _, _, test_set = split_dataset(dataset, seed=42)
    dataloader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)

    evaluate(model, dataloader, device, 
             output_dir=args.output_dir, 
             save_audio=args.save_audio, 
             save_every=args.save_every, 
             snr_thresh=args.snr_thresh)