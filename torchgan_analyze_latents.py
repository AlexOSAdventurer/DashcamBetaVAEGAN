import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
BCE_logits = F.binary_cross_entropy_with_logits
import torch.nn as nn
import torch.distributed as dist
import torchgan
import builtins
import sys
import random
import gc
import matplotlib
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.autograd import grad as torch_grad

torch.autograd.set_detect_anomaly(True)

load_prefix = str((("LOAD_PREFIX" in os.environ) and os.environ["LOAD_PREFIX"]) or "")

zDim=512
zDim_travel = 512
zDim_image_step_size = 32
mean_step_size = 10.0
mean_step_count = 11
imageIndex = 7571
device="cuda"

plt.rcParams["figure.figsize"] = [7.5*mean_step_count, 11*zDim_image_step_size]
plt.rcParams["figure.autolayout"] = True

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class ImageDataset(Dataset):
    def __init__(self, images_path):
        self.images_data = np.load(images_path, mmap_mode='r')
        self.total_sequences = self.images_data.shape[0]
        print(self.total_sequences)

    def __getitem__(self, index):
        return torch.clamp(((torch.from_numpy(self.images_data[index].copy()).type(torch.FloatTensor) / 255.0) - 0.5) * 2.0, -1.0, 1.0)

    def __len__(self):
        return self.total_sequences
        
#256x5x10
class Encoder(nn.Module):
    def __init__(self, imgChannels=3, featureDim=zDim*4*4, zDim=zDim):
        super(Encoder, self).__init__()
        self.featureDim = featureDim

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Sequential(nn.Conv2d(imgChannels, 64, 5, stride=(2,2), padding=2),  nn.LeakyReLU(0.2))
        self.encConv2 = nn.Sequential(nn.Conv2d(64, 128, 5, stride=(2,2), padding=2), nn.LeakyReLU(0.2))
        self.encConv3 = nn.Sequential(nn.Conv2d(128, 256, 5, stride=(2,2), padding=2), nn.LeakyReLU(0.2))
        self.encConv4 = nn.Sequential(nn.Conv2d(256, 512, 5, stride=(2,2), padding=2), nn.LeakyReLU(0.2))
        self.encConv5 = nn.Sequential(nn.Conv2d(512, zDim, 5, stride=(2,2), padding=2), nn.LeakyReLU(0.2), nn.Flatten())
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Sequential(nn.Linear(featureDim, zDim))

    def encoder(self, x):
        x = self.encConv1(x)
        x = self.encConv2(x)
        x = self.encConv3(x)
        x = self.encConv4(x)
        x = self.encConv5(x)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        return self.encoder(x)
        
#256x5x10
class Decoder(torchgan.models.DCGANGenerator):
    def __init__(self, size=128):
        super(Decoder, self).__init__(encoding_dims=zDim, out_size=size, batchnorm=False)
        
    def forward(self, x, feature_matching=False):
        return super(Decoder, self).forward(x, feature_matching)

dataset_val = ImageDataset("data/val_float_128x128.npy")
val_sampler = torch.utils.data.RandomSampler(dataset_val, replacement=True, num_samples=1)
val_loader = torch.utils.data.DataLoader(dataset_val,batch_size=1, sampler=val_sampler)
torch.backends.cudnn.benchmark = True

encoder = Encoder().cuda()
decoder = nn.SyncBatchNorm.convert_sync_batchnorm(Decoder()).cuda()

if (load_prefix):
    encoder.load_state_dict(torch.load(load_prefix + "_encoder.model"))
    decoder.load_state_dict(torch.load(load_prefix + "_decoder.model"))

def test_generation_step(imgs):
    encoder.eval()
    decoder.eval()

    print(imgs.shape)
    E_mean, E_var = encoder.forward(imgs)
    E_mean_total = torch.empty((zDim_travel * mean_step_count, zDim), device = imgs.device)
    print(E_mean.shape)
    for i in range(zDim_travel):
        for j in range(mean_step_count):
            E_mean_new = E_mean.clone().reshape((zDim))
            E_mean_new[i] = (E_mean_new[i] + ((j - (mean_step_count // 2)) * mean_step_size))
            E_mean_total[(i * mean_step_count) + j] = E_mean_new
            
    G_imgs = decoder.forward(E_mean_total)
    print(G_imgs.shape)
    G_imgs = G_imgs.reshape((zDim_travel, mean_step_count, 3, 128, 128))
    print(G_imgs.shape)
    return G_imgs

def save_images(out, path):
    fig = plt.Figure()
    zDim = out.shape[0]
    step_position = out.shape[1]
    for a in range(0, zDim, zDim_image_step_size):
        for i in range(zDim_image_step_size):
            for j in range(step_position):
                img = (np.transpose(out[a + i][j].detach().cpu().numpy(), [1,2,0]) + 1) / 2.0
                ax1 = plt.subplot(zDim_image_step_size, mean_step_count, (i * mean_step_count) + j + 1)
                ax1.imshow(np.squeeze(img))
        newpath = path + "_zDimIndex"+str(a)+"_imageIndex"+str(imageIndex)+".png"
        plt.savefig(newpath)

"""
Training the network for a given number of epochs
The loss after every epoch is printed
"""
print("Launching!!")
sys.stdout.flush()

with torch.no_grad():
    imgs = dataset_val[imageIndex].reshape((1, 3, 128, 128))
    imgs = imgs.to(device)
    out = test_generation_step(imgs)
    save_images(out, "latent_analysis/" + load_prefix)
