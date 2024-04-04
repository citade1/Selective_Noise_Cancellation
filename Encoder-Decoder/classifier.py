# classifier after encoder 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import copy 
import tqdm

from Autoencoder import AutoEncoder
from getDataLoader import getDataLoader 
import Autoencoder as ae

train_loader, test_loader = getDataLoader()

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # self.modelB = model
        self.fc1 = nn.Linear(10, 4)
        self.fc2 = nn.Linear(4, 2)
        # self.fc2 = nn.Linear(1024,128)
        # self.out = nn.Linear(128, 2)
        # self.drop = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # x = x.view(1, 32*16*16)
        x = torch.relu(self.fc1(x))
        # x = self.drop(x)
        # x = torch.relu(self.fc2(x))
        # x = self.drop(x)
        # x = self.out(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

model = AutoEncoder()
classification_model = Classifier()
_, model = ae.train(tl = train_loader, ml = model)

# Optimizer and Loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr= 0.001)
# Decay LR by a factor of 0.1 every 7 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=3, verbose=True)

copied_model = copy.deepcopy(model)

from ast import If
model_filename = 'model_classfier.pt'
n_epochs = 10

train_losses=[]
train_acc = []

for epoch in range(1, n_epochs+1):
    train_loss = 0
    running_acc = 0
    classification_model.train()
    for x, label in tqdm(train_loader):
        # optimizer.zero_grad() #기울기에 대한 정보를 초기화합니다.
        
        encoded, decoded = model(x)
        y_pred = classification_model(encoded)

        zz = torch.zeros(1,2)
        if label.item() == 0:
          zz[0][0] = 1
        else:
          zz[0][1] = 1

        loss = criterion(y_pred, zz) # decoded와 원본이미지(y) 사이의 평균제곱오차를 구합니다

        loss.backward() # 기울기를 구합니다.
        
        optimizer.step() #최적화를 진행합니다.

        train_loss += loss.item()
    
    
    # calculate mean for each batch
    train_losses.append(train_loss / len(train_loader))
    train_acc.append(running_acc / len(train_loader))
    torch.save(model.state_dict(), model_filename)
    print("Epoch:{}/{}..".format(epoch, n_epochs),
            "Train Loss: {:.3f}..".format(train_loss / len(train_loader)))

# print whether user is in the subway or not 
# 0: subway 1: other environment
if y_pred[0][0]>=0.5:
    print("User is on the subway")
else:
    print("User is in other environment")

# print test accuracy of this classifier
classification_model.eval()
total_correct = 0
with torch.no_grad():
  for images, labels in test_loader:
      encoded, decoded = model(images)
      output = classification_model(encoded)

      _, pred = torch.max(output, 1)
      total_correct += torch.sum(pred == labels).item()
  accuracy = total_correct / len(test_loader.dataset)
  print('Test Accuracy: {}'.format(accuracy))




        
