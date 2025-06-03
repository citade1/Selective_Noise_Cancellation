import os
import torch
import matplotlib.pyplot as plt


def tensor_to_img(spec, out_path, cmap="magma"):

    # Load tensor (expected shape: [1,F,T] or [F,T])
    if spec.dim()==3:
        spec = spec.squeeze(0)
    
    # plot
    plt.figure(figsize=(10, 4))
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.axis("auto")
    plt.imshow(spec.cpu().numpy(), origin="lower", aspect="auto", cmap=cmap)

    # save as png
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"Saved spectrogram image to: {out_path}")