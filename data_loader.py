import numpy as np
from torchvision import transforms
from utils import *
import torch
from torch.utils import data
import pandas as pd
import numpy as np
from skimage import io
from torch.utils.data import TensorDataset


def compute_normalization(loader):

    mean, std = online_mean_and_sd(loader)
    
    print(mean, std)
    
    
def get_data_loader(batch_size, is_train=True):

    print('Loading FER2013 dataset')

    if is_train:
        paths = ['/content/drive/MyDrive/PFG/data/preprocessed_train.csv',
                '/content/drive/MyDrive/PFG/data/preprocessed_val.csv']
    else:
        paths = ['/content/drive/MyDrive/PFG/data/preprocessed_test.csv']
    
    dataloaders = []

    for path in paths:
        print(f'Loading from {path}')
        df = pd.read_csv(path)
        imgs = [path[25:] for path in df['img']]
        labels = [label for label in df['emotion']]
        input = zip(imgs,labels)

        images = []
        labels = []
        for X, Y in input:
            splitted = X.split('/')
            X = f'/content/{splitted[-2]}/{splitted[-1]}'
            img = io.imread(X , as_gray=True)
            
            images.append(np.array(img))
            label = np.array(Y)
            labels.append(label)
        images = np.array(images)
        labels = np.array(labels)

        tensor_x = torch.Tensor(images) # transform to torch tensor
        tensor_y = torch.Tensor(labels)

        dataset = TensorDataset(tensor_x, tensor_y)
        dataloaders.append(data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True))

    return dataloaders


if __name__ == "__main__":
       
    # Load the samples
    train_loader, _, _ = get_data_loader(128)
    
    compute_normalization(train_loader)