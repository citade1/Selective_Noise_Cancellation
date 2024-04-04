# latent space visualization
import matplotlib.pyplot as plt
from pandas.core.common import flatten
import copy
import numpy as np
import random
import glob
import math 

import torch
from torch import nn
from torch import optim
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm

from sklearn.manifold import TSNE
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D #생성되는 이미지를 관찰하기 위함입니다. 3차원 플롯을 그리는 용도입니다.
from matplotlib import cm # 데이터포인트에 색상을 입히는 것에 사용됩니다.

from getDataLoader import getDataLoader
from Autoencoder import AutoEncoder
import Autoencoder as ae

train_subway_loader, train_other_loader = getDataLoader('train/subway/','train/other/')

# start visualization
tsne = TSNE()

class_obj = {
    0: 'subway',
    1: 'other',
}

train_loader, test_loader = getDataLoader()
model = AutoEncoder()
_, model = ae.train(ml = model, tl = train_loader)
model.eval()

ses = []
oes = []
if __name__ == '__main__':
  for images, labels in train_subway_loader:
    subway_encoded, subway_decoded = model(images)
    ses.append(subway_encoded[0].detach().cpu().numpy())
  for images, labels in train_other_loader:
    other_encoded, other_decoded = model(images)
    oes.append(other_encoded[0].detach().cpu().numpy())

ses = np.array(ses)

tsne_subway_v = TSNE(n_components=3).fit_transform(ses)

oes = np.array(oes)

tsne_other_v = TSNE(n_components=3).fit_transform(oes)


print("tsne_other_v")
type_tf=['other', 'subway']

# plot in 3d space

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')

for t in tsne_other_v:
  ax.scatter(t[0], t[1], t[2], c='blue')
for s in tsne_subway_v:
  ax.scatter(s[0], s[1], s[2], c='green')


ax.legend(labels=type_tf)
plt.show()
