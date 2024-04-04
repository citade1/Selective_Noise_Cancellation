# Autoencoder !! 
import torch
from torch import nn
from torch import optim
import copy
from getDataLoader import getDataLoader

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias = False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias = False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias = False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias = False),
            # nn.ReLU(),
            # nn.MaxPool2d(2,2)
          )
        
        # decoder
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(128, 128, kernel_size = 2, stride = 2, padding=0, bias = False),
            # nn.ReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size = 2, stride = 2, padding=0, bias = False),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, kernel_size = 2, stride = 2, padding=0, bias = False),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, kernel_size = 2, stride = 2, padding=0, bias = False),
            nn.Sigmoid())
        
        self.dense = nn.Linear(16*16*32, 10)
        self.broaden = nn.Linear(10, 16*16*32)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(x.shape[0], 16*16*32)
        encoded = self.dense(encoded)
        broaded = self.broaden(encoded)
        broaded = broaded.reshape(x.shape[0], 32,16,16)
        decoded = self.decoder(broaded)

        return encoded, decoded

model = AutoEncoder()

init_weights = copy.deepcopy(model.encoder[0].weight.data)

optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
criterion = nn.MSELoss()

train_loader, test_loader = getDataLoader()
def train(tl=train_loader, ml = model, epochs = 40):
    
    train_loss = []
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        n = 1
        runningloss = 0
        if __name__ == '__main__':
            for images, labels in train_loader:
                optimizer.zero_grad()
                encoded, output = model(images)
                loss = criterion(output, images)
                loss.backward()
                optimizer.step()
                runningloss += loss.item() / images.shape[0]
                train_loss.append(runningloss)
                n += 1
            print('Epoch: {}/{} \t Mean Square Error Loss: {}'.format(epoch + 1, epochs, runningloss))
    return train_loss, model
