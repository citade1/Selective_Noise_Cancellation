import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from models import UNet, AttentionUNet, StackedTransformerEncoder
from utils import waveform_to_spectrogram, mix_audio, load_audio, split_dataset
from data_preparation import SpectrogramDataset
from datetime import datetime
import argparse
from tqdm import tqdm

# Colab override for argpase 
# import sys
# sys.argv = ['train.py', '--model', 'attention', '--epochs', '10', '--batch_size', '8', '--optimizer', 'adamw']

# -----------------
# Evaluation Metric
# -----------------
def snr(pred, target):
    signal_power = torch.mean(target**2)
    noise_power = torch.mean((target - pred)**2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    return snr


# ----------------
# Traning Function
# ----------------
def train(model, dataloader, criterion, optimizer, scheduler, device, epoch, writer, val_loader, best_val_loss, save_path):
    model.train()
    running_loss, running_snr = 0.0, 0.0

    for i, (mixture, target) in enumerate(tqdm(dataloader, desc="Training")):
        mixture, target = mixture.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(mixture)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_snr += snr(output, target).item()

        if i%10==0:
            print(f"Batch {i+1}/{len(dataloader)}, Loss:{loss.item():.4f}, SNR: {snr(output, target):.2f} dB")

    avg_loss = running_loss / len(dataloader) 
    avg_snr = running_snr / len(dataloader)  

    # Validation
    model.eval()
    val_loss, val_snr = 0.0, 0.0

    with torch.no_grad():
        for mixture, target in val_loader:
            mixture, target = mixture.to(device), target.to(device)
            output = model(mixture)
            val_loss += criterion(output, target).item()
            val_snr += snr(output, target).item()
    
    val_loss /= len(val_loader)
    val_snr /= len(val_loader)

    scheduler.step(val_loss)

    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("SNR/train", avg_snr, epoch)
    writer.add_scalar("SNR/val", val_snr, epoch)

    print(f"Epoch Summary - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, SNR: {val_snr:.2f} dB")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Best model updated and saved to {save_path}")

    return best_val_loss


# -----------
# Main script
# -----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['unet', 'attention'], default='unet')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd'], default='adam')
    parser.add_argument('--save_dir', type=str, default="checkpoints")
    parser.add_argument('--snr_min', type=float, default=5.0)
    parser.add_argument('--snr_max', type=float, default=15.0)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model configuration
    if args.model == "unet":
        model = UNet(in_channels=1, out_channels=1).to(device)
    elif args.model == "attention":
        model = AttentionUNet(
            in_channels=1,
            out_channels=1,
            features = [64, 128, 256],
            attention_block=StackedTransformerEncoder,
            attn_kwargs = {"d_model": 256, "n_heads": 4, "dim_ff": 512, "num_layers": 1}
        ).to(device)
    
    # Dataset and Dataloader
    dataset = SpectrogramDataset(
        target_dir="data/targets",
        noise_dirs=["data/noises/fsd50k_clips", "data/noises/freesound_misc_clips"],
        snr_range=(args.snr_min, args.snr_max),
        n_noises=3
    )
    train_set, val_set, _ = split_dataset(dataset, seed=42)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    # Loss
    criterion = nn.MSELoss()

    # Optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer =="sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Resume from checkpoint if exists
    best_val_loss = float('inf')
    save_path = os.path.join(args.save_dir, f"{args.model}_best.pt")

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(f"Resumed training from checkpoint: {save_path}")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        best_val_loss = train(model, train_loader, criterion, optimizer, scheduler, device, 
                              epoch, writer, val_loader, best_val_loss, save_path)
    
    writer.close()
