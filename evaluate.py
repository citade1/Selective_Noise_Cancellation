import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import UNet, AttentionUNet, StackedTransformerEncoder
from data_preparation import SpectrogramDataset
from utils import spectrogram_to_waveform, split_dataset, tensor_to_img
from utils.dnsmos_infer import run_dnsmos
from utils.csv_logger import CSVLogger
import argparse
from datetime import datetime


DEFAULT_SR = 16000
LOG_DIR = "runs"
os.makedirs(LOG_DIR, exist_ok=True)


@torch.no_grad()
def evaluate(model, dataloader, device, output_wav_dir, output_spec_dir, model_name, ovr_thresh, bak_thresh, save=False):
    model.eval()
    os.makedirs(output_wav_dir, exist_ok=True)
    os.makedirs(output_spec_dir, exist_ok=True)
    
    total_samples = 0
    total_sig, total_bak, total_ovr = 0, 0, 0

    # per-sample logging file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    DNSMOS_LOG_FILE = f"{model_name}_DNSMOS_persample_log.csv"
    dnsmos_logger = CSVLogger(
        file_path=os.path.join(LOG_DIR, DNSMOS_LOG_FILE),
        fieldnames=["timestamp", "file_path", "SIG", "BAK", "OVR", "saved", "reason"]
    )
    
    # go through batch 
    for batch_idx, (mixture_spec, _, mixture_phase, length) in enumerate(tqdm(dataloader, desc="Inference")):
        mixture_spec, mixture_phase = mixture_spec.to(device), mixture_phase.to(device)
        est_spec = model(mixture_spec) # denoised spectrogram

        batch_size = est_spec.size(0)
        for i in range(batch_size):
            sample_length = int(length[i]) 
            total_samples += 1

            est_wave_i = spectrogram_to_waveform(est_spec[i], phase=mixture_phase[i], length=sample_length)
            mix_wave_i = spectrogram_to_waveform(mixture_spec[i], phase=mixture_phase[i], length=sample_length)

            # save denoised audio temporarily for DNSMOS
            est_path = os.path.join(output_wav_dir, f"sample_{batch_idx}_{i}_denoised.wav")
            torchaudio.save(est_path, est_wave_i.cpu(), DEFAULT_SR)

            # Evaluate quality of an audio sample
            dns_scores = run_dnsmos(est_path)
            sig, bak, ovr = dns_scores["SIG"], dns_scores["BAK"], dns_scores["OVR"]

            total_sig += sig
            total_bak += bak
            total_ovr += ovr

            # logging
            should_save = save and (ovr >= ovr_thresh or bak <= bak_thresh)
            reason = "both" if ovr >= ovr_thresh and bak <= bak_thresh else \
                    "ovr" if ovr >= ovr_thresh else \
                    "bak" if bak <= bak_thresh else ""
            saved_flag = "yes" if reason else "no"

            dnsmos_logger.log({
                "timestamp": timestamp,
                "file_path": est_path,
                "SIG": sig,
                "BAK": bak,
                "OVR": ovr,
                "saved": saved_flag,
                "reason": reason
            })

            if should_save:
                torchaudio.save(os.path.join(output_wav_dir, f"sample_{batch_idx}_{i}_mix.wav"),
                                mix_wave_i.cpu(), DEFAULT_SR)

                tensor_to_img(mixture_spec[i].cpu(), os.path.join(output_spec_dir, f"sample_{batch_idx}_{i}_mix.png"))
                tensor_to_img(est_spec[i].cpu(), os.path.join(output_spec_dir, f"sample_{batch_idx}_{i}_denoised.png"))
            else:
                os.remove(est_path)

    dnsmos_logger.close()

    avg_sig = total_sig / total_samples
    avg_bak = total_bak / total_samples
    avg_ovr = total_ovr / total_samples

    TEST_LOG_FILE = f"{model_name}_test_log.csv"
    test_logger = CSVLogger(
        file_path=os.path.join(LOG_DIR, TEST_LOG_FILE),
        fieldnames=["timestamp", "Avg_SIG", "Avg_BAK", "Avg_OVR"]
    )
    test_logger.log({
        "timestamp":timestamp,
        "Avg_SIG": avg_sig,
        "Avg_BAK": avg_bak,
        "Avg_OVR": avg_ovr
    })
    test_logger.close()
    
    print(f"\n [{model_name}] Average metrics over {total_samples} samples: "
      f"Speech quality: {avg_sig:.2f} | Background noise intrusiveness: {avg_bak:.2f} | Overall quality: {avg_ovr:.2f}")

# -----------
# Main Script
# -----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['unet', 'attention'], default='unet')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='eval_outputs')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--noise_fsd_dir', type=str, required=True)
    parser.add_argument('--noise_misc_dir', type=str, required=True)
    parser.add_argument('--snr_min', type=float, default=-5.0)
    parser.add_argument('--snr_max', type=float, default=5.0)
    parser.add_argument('--n_noises', type=int, default=3)
    parser.add_argument('--ovr_thresh', type=float, default=4.0)
    parser.add_argument('--bak_thresh', type=float, default=2.0)
    args = parser.parse_args()

    # device
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
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Support both full dict checkpoints and raw state_dicts
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded model_state from checkpoint dictionary: {args.checkpoint}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Loaded raw model state_dict from: {args.checkpoint}")

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

    # evaluate
    evaluate(model, dataloader, 
             device=device,
             output_wav_dir=os.path.join(args.output_dir, f"{args.model}", "wav"),
             output_spec_dir=os.path.join(args.output_dir, f"{args.model}","spec"),
             model_name=args.model,
             ovr_thresh=args.ovr_thresh,
             bak_thresh=args.bak_thresh,
             save=args.save)