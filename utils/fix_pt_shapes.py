import os
import torch
from tqdm import tqdm
import argparse

def fix_waveform_shape(waveform):
    if not isinstance(waveform, torch.Tensor):
        raise TypeError("Expected a torch.Tensor")
    
    waveform = waveform.squeeze()

    if waveform.dim()==1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() == 2 and waveform.shape[0] > 1:  # stereo
        waveform = waveform.mean(dim=0, keepdim=True)   # convert to mono with shape (1, T)
    if waveform.dim() != 2 or waveform.shape[0] != 1:
        raise ValueError(f"Unexpected shape after squueze: {waveform.shape}")
    
    return waveform

def fix_all_pt_files(root_dir):
    fixed = 0
    skipped = 0

    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.startswith("._") and not fname.endswith(".pt"):
                continue
                
            path = os.path.join(root,fname)
            try:
                tensor = torch.load(path)
                if not isinstance(tensor, torch.Tensor):
                    continue

                fixed_tensor = fix_waveform_shape(tensor)

                if tensor.shape != fixed_tensor.shape:
                    torch.save(fixed_tensor, path)
                    fixed += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"[Error] {path}:{e}")
    print(f"\n Done, Fixed: {fixed}, Skipped: {skipped}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        raise FileNotFoundError(f"Directory not found: {args.dir}")
    fix_all_pt_files(args.dir)