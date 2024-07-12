import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
from transformer_sound_separator import TransformerSoundSeparator

# Paths to be updated
load_path = '/content/drive/MyDrive/sncWhisper/spectrograms/'

# Load the spectrograms
mixture_spectrograms = np.load(os.path.join(load_path, 'mixture_spectrograms.npy'))
source_spectrograms = np.load(os.path.join(load_path, 'source_spectrograms.npy'))

class SoundSeparationDataset(Dataset):
    def __init__(self, mixtures, sources):
        self.mixtures = mixtures
        self.sources = sources

    def __len__(self):
        return len(self.mixtures)

    def __getitem__(self, idx):
        mixture = torch.tensor(self.mixtures[idx], dtype=torch.float32)
        source = torch.tensor(self.sources[idx], dtype=torch.float32)
        return mixture, source

# Create dataset and dataloader
dataset = SoundSeparationDataset(mixture_spectrograms, source_spectrograms)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model training
def train_model(model, train_loader, num_epochs=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for mixtures, sources in train_loader:
            optimizer.zero_grad()
            outputs = model(mixtures)
            loss = criterion(outputs, sources)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

# Initialize the model
model = TransformerSoundSeparator()
# Train the model
train_model(model, train_loader, num_epochs=10)
# Save the trained model
torch.save(model.state_dict(), 'sncWhisper_sound_separator.pth')
