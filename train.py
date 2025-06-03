import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import UNet, AttentionUNet, StackedTransformerEncoder
from utils import split_dataset, snr
from utils.csv_logger import CSVLogger
from utils.early_stopping import EarlyStopping
from data_preparation import SpectrogramDataset
import argparse
from tqdm import tqdm
from datetime import datetime

LOG_DIR = "runs"
os.makedirs(LOG_DIR, exist_ok=True)

# ---------
# one epoch
# ---------

def run_epoch(model, dataloader, criterion, optimizer, device, mode="train"):
    model.train() if mode=="train" else model.eval()
    total_loss, total_snr = 0.0, 0.0

    for mixture, target, _, _ in tqdm(dataloader, desc=f"{mode.capitalize()}"):
        mixture, target = mixture.to(device), target.to(device)

        if mode=="train":
            optimizer.zero_grad()
        output = model(mixture)
        loss = criterion(output, target)
        if mode=="train":
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_snr += snr(output, target).item()
    
    n = len(dataloader)
    return total_loss/n, total_snr/n



# -----------
# Main script
# -----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['unet', 'attention'], default='unet')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd'], default='adam')
    parser.add_argument('--save_dir', type=str, default="checkpoints")
    parser.add_argument('--snr_min', type=float, default=-5.0)
    parser.add_argument('--snr_max', type=float, default=5.0)
    parser.add_argument('--n_noises', type=int, default=3)
    parser.add_argument('--earlystop_patience', type=int, default=2)
    parser.add_argument('--target_dir', type=str, required=True) # input directory for target
    parser.add_argument('--noise_fsd_dir', type=str, required=True) # input directory for noises (fsd50k dataset)
    parser.add_argument('--noise_misc_dir', type=str, required=True) # input directory for misc noises
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    # device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # mac os 

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
        target_dir=args.target_dir,
        noise_fsd_dir=args.noise_fsd_dir,
        noise_misc_dir=args.noise_misc_dir,
        snr_range=(args.snr_min, args.snr_max),
        n_noises=args.n_noises
    )
    train_set, val_set, _ = split_dataset(dataset, seed=42)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)
    
    # Loss
    criterion = nn.MSELoss()

    # Optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer =="sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    # Resume from checkpoint if exists
    best_val_loss = float('inf')
    save_path = os.path.join(args.save_dir, f"{args.model}_best_checkpoint.pt")
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"Resumed training from checkpoint: {save_path}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    TRAIN_LOG_FILE = f"train_log_{args.model}.csv"
    log_file_path = os.path.join(LOG_DIR, TRAIN_LOG_FILE)
    
    early_stopper = EarlyStopping(patience=args.earlystop_patience)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        # train
        train_loss, train_snr = run_epoch(model, train_loader, criterion, optimizer, device, mode="train")
        # validation
        val_loss, val_snr = run_epoch(model, val_loader, criterion, None, device, mode="validation")

        # adapt learning rate
        scheduler.step(val_loss)

        # logging
    
        train_logger = CSVLogger(
            file_path=log_file_path,
            fieldnames=["timestamp", "model", "epoch", "train_loss", "val_loss", 
                        "train_snr", "val_snr", "lr", "snr_min", "snr_max", "n_noises"]
        )
        train_logger.log({
            "timestamp":timestamp,
            "model": args.model,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_snr": train_snr,
            "val_snr": val_snr,
            "lr": optimizer.param_groups[0]['lr'],
            "snr_min": args.snr_min,
            "snr_max": args.snr_max,
            "n_noises": args.n_noises
        })
        train_logger.close()
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
      f"Train SNR: {train_snr:.2f}dB, Val SNR: {val_snr:.2f}dB")

        # save model if val_loss has been improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "best_val_loss": val_loss,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }
            torch.save(checkpoint, save_path)
            print(f"Best model updated and saved to {save_path}")

        early_stopper(val_loss)
        if early_stopper.early_stop:
            print(f"Early stopping triggered after {epoch+1}epochs")
            break
        
        
        
    
