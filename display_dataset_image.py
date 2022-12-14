import os
import numpy as np
import builtins
import sys
import random
import torch
import gc
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

plt.rcParams["figure.figsize"] = [7.5, 11]
plt.rcParams["figure.autolayout"] = True

index = 7571

class ImageDataset(Dataset):
    def __init__(self, images_path):
        self.images_data = np.load(images_path, mmap_mode='r')
        self.total_sequences = self.images_data.shape[0]
        print(self.total_sequences)

    def __getitem__(self, index):
        return torch.clamp(((torch.from_numpy(self.images_data[index].copy()).type(torch.FloatTensor) / 255.0) - 0.5) * 2.0, -1.0, 1.0)

    def __len__(self):
        return self.total_sequences

dataset_val = ImageDataset("data/val_float_128x128.npy")

def save_image(out, path):
    fig = plt.Figure()
    img = (np.transpose(out[0].detach().cpu().numpy(), [1,2,0]) + 1) / 2.0
    ax1 = plt.subplot(1, 1, 1)
    ax1.imshow(np.squeeze(img))
    plt.savefig(path)

with torch.no_grad():
    imgs = dataset_val[index].reshape((1, 3, 128, 128))
    save_image(imgs, "latent_analysis/"+str(index)+".png")