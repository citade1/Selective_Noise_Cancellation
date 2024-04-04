# label image and return dataset as dataloader

import matplotlib.pyplot as plt
from pandas.core.common import flatten
import copy
import numpy as np
import random

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision

from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset

import cv2

import glob
from tqdm import tqdm

def getDataLoader(datapath1='train/', datapath2='test/'):
    root = 'aeDataset/'
    train_data_path = root + datapath1
    test_data_path = root + datapath2

    train_image_paths = [] #to store image paths in list
    classes = [] #to store class values


    # get all the paths from train_data_path and append image paths and class to respective lists
    for data_path in glob.glob(train_data_path + '/*'):
        classes.append(data_path.split('/')[-1]) 
        train_image_paths.append(glob.glob(data_path + '/*'))
    
    train_image_paths = list(flatten(train_image_paths))
    random.shuffle(train_image_paths)

    # create the test_image_paths
    test_image_paths = []
    for data_path in glob.glob(test_data_path + '/*'):
        test_image_paths.append(glob.glob(data_path + '/*'))

    test_image_paths = list(flatten(test_image_paths))

    idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}


    class SubwayDataset(Dataset):
        def __init__(self, image_paths, transform = None):
            self.image_paths = image_paths
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_filepath = self.image_paths[idx]
            image = cv2.imread(image_filepath,0)
            image = torchvision.transforms.functional.to_tensor(cv2.resize(image, dsize=(128,128)))
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
            label = image_filepath.split('/')[-2]
            label = class_to_idx[label]
            if self.transform is not None:
                image = self.transform(image=image)["image"]
            return image, label


    train_dataset = SubwayDataset(train_image_paths)
    test_dataset = SubwayDataset(test_image_paths)

    train_dataset, test_dataset
    
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )
    return train_loader, test_loader
