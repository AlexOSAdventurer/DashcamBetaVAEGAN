import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(int(os.environ["SLURM_PROCID"]) % 2)
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


plt.rcParams["figure.figsize"] = [15.0, 350.0]
plt.rcParams["figure.autolayout"] = True

batch_size = 64
learning_rate = 2e-5
betas = (0.5, 0.999)
load_prefix = str((("LOAD_PREFIX" in os.environ) and os.environ["LOAD_PREFIX"]) or "")
save_prefix = str((("SAVE_PREFIX" in os.environ) and os.environ["SAVE_PREFIX"]) or "")

num_epochs = 1000
zDim=512
device="cuda"

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["SLURM_PROCID"])
vae_kl_weight = float(os.environ["VAE_KL_WEIGHT"])
print(world_size, rank, vae_kl_weight)

class ImageDataset(Dataset):
    def __init__(self, images_path):
        self.images_data = np.load(images_path, mmap_mode='r')
        self.total_sequences = self.images_data.shape[0]
        print(self.total_sequences)

    def __getitem__(self, index):
        return torch.clamp(((torch.from_numpy(self.images_data[index].copy()).type(torch.FloatTensor) / 255.0) - 0.5) * 2.0, -1.0, 1.0)

    def __len__(self):
        return self.total_sequences

class VAE_Discriminator(torchgan.models.DCGANDiscriminator):
    def __init__(self, size=128):
        super(VAE_Discriminator, self).__init__(in_size=size, batchnorm=False)
        #self.instanceNorm = nn.InstanceNorm2d((3,size,size))
        self.instanceNorm = nn.Identity()

    def forward(self, x, feature_matching=False):
        return super(VAE_Discriminator, self).forward(self.instanceNorm.forward(x), feature_matching)

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

class Decoder(torchgan.models.DCGANGenerator):
    def __init__(self, size=128):
        super(Decoder, self).__init__(encoding_dims=zDim, out_size=size, batchnorm=False)
        
    def forward(self, x, feature_matching=False):
        return super(Decoder, self).forward(x, feature_matching)

if rank != 0:
	def print_pass(*args):
		pass
	builtins.print = print_pass
else:
    print(open(__file__, 'r').read())

print("Starting process group!")
sys.stdout.flush()
dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
print("Started!")
sys.stdout.flush()

dataset_train = ImageDataset("data/train_float_128x128.npy")
dataset_val = ImageDataset("data/val_float_128x128.npy")

train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
val_sampler = torch.utils.data.RandomSampler(dataset_val, replacement=True, num_samples=16)
train_loader = torch.utils.data.DataLoader(dataset_train,batch_size=batch_size, num_workers=30, pin_memory=False, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(dataset_val,batch_size=16, sampler=val_sampler)
print("Dataloaders ready!")
sys.stdout.flush()
torch.backends.cudnn.benchmark = True

"""
Initialize the network and the Adam optimizer
"""
encoder = Encoder().cuda()
decoder = nn.SyncBatchNorm.convert_sync_batchnorm(Decoder()).cuda()
discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(VAE_Discriminator()).cuda()

print("Network ready!")
sys.stdout.flush()

if (load_prefix):
    encoder.load_state_dict(torch.load(load_prefix + "_encoder.model"))
    decoder.load_state_dict(torch.load(load_prefix + "_decoder.model"))
    discriminator.load_state_dict(torch.load(load_prefix + "_discriminator.model"))

encoder = torch.nn.parallel.DistributedDataParallel(encoder, find_unused_parameters=False)
decoder = torch.nn.parallel.DistributedDataParallel(decoder, find_unused_parameters=False)
discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, find_unused_parameters=True)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, betas=betas)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, betas=betas)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=betas)


def kl_divergence_func(mean, var):
    #var = var.clamp(-100, 15.0)
    return torch.mean(-var + (0.5 * (-1 + torch.exp(2.0 * var) + torch.square(mean))))

def encoder_train_step(imgs, batch_size):
    E_mean, E_var = encoder.forward(imgs)
    E_reparam = encoder.module.reparameterize(E_mean, E_var)
    G_imgs = decoder.forward(E_reparam)

    D_imgs_activation = discriminator.forward(imgs, feature_matching=True)
    D_decoder_activation = discriminator.forward(G_imgs, feature_matching=True)

    kl_divergence = kl_divergence_func(E_mean, E_var) * vae_kl_weight
    like_loss = torch.nn.functional.mse_loss(D_imgs_activation, D_decoder_activation)
    encoder_loss = kl_divergence + like_loss

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()
    encoder_loss.backward()
    encoder_optimizer.step()

    shared = torch.empty(2, dtype=torch.float)
    shared[0] = float(kl_divergence)
    shared[1] = float(like_loss)
    shared = shared.cuda()
    dist.all_reduce(shared, op=dist.ReduceOp.SUM)
    shared = shared.cpu()


    return float(shared[0] / float(world_size)), float(shared[1] / float(world_size))

def decoder_train_step(imgs, batch_size):
    E_mean, E_var = encoder.forward(imgs)
    E_reparam = encoder.module.reparameterize(E_mean, E_var).detach()
    fake_mean = torch.randn_like(E_reparam)
    G_imgs = decoder.forward(E_reparam)
    G_fake = decoder.forward(fake_mean)

    D_imgs_activation = discriminator.forward(imgs, feature_matching=True)
    D_decoder_activation = discriminator.forward(G_imgs, feature_matching=True)
    D_decoder_logit = discriminator.forward(G_imgs)
    D_fake_logit = discriminator.forward(G_fake)

    decoder_loss_encoded = torch.mean(D_decoder_logit)#torch.mean(BCE_logits(D_decoder_logit, torch.ones_like(D_decoder_logit)))
    decoder_loss_fake = torch.mean(D_fake_logit)#torch.mean(BCE_logits(D_fake_logit, torch.ones_like(D_fake_logit)))
    like_loss = torch.nn.functional.mse_loss(D_imgs_activation, D_decoder_activation) #+ (torch.nn.functional.l1_loss(imgs, G_imgs) * 255.0)
    decoder_loss = (decoder_loss_encoded / 1.0) + (decoder_loss_fake / 1.0) + like_loss

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()
    decoder_loss.backward()
    decoder_optimizer.step()

    shared = torch.empty(3, dtype=torch.float)
    shared[0] = float(decoder_loss_encoded)
    shared[1] = float(decoder_loss_fake)
    shared[2] = float(like_loss)
    shared = shared.cuda()
    dist.all_reduce(shared, op=dist.ReduceOp.SUM)
    shared = shared.cpu()


    return float(shared[0] / float(world_size)), float(shared[1] / float(world_size)), float(shared[2] / float(world_size))

def discriminator_train_step(imgs, batch_size, train):
    E_mean, E_var = encoder.forward(imgs)
    E_reparam = encoder.module.reparameterize(E_mean, E_var)
    fake_mean = torch.randn_like(E_reparam)
    G_imgs = decoder.forward(E_reparam)
    G_fake = decoder.forward(fake_mean)

    D_imgs_logit = discriminator.forward(imgs)
    D_decoder_logit = discriminator.forward(G_imgs)
    D_fake_logit = discriminator.forward(G_fake)

    discriminator_loss_imgs = torch.mean(D_imgs_logit)#torch.mean(BCE_logits(D_imgs_logit, torch.ones_like(D_imgs_logit)))
    discriminator_loss_decoder = -0.5 * torch.mean(D_decoder_logit)#torch.mean(BCE_logits(D_decoder_logit, torch.zeros_like(D_decoder_logit)))
    discriminator_loss_fake = -0.5 * torch.mean(D_fake_logit)#torch.mean(BCE_logits(D_fake_logit, torch.zeros_like(D_fake_logit)))
    #GP loss
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(imgs)
    alpha = alpha.cuda()
    interpolated = alpha * imgs.data + (1 - alpha) * G_imgs.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda()
    # Calculate rating of interpolated samples
    D_interp_logit = discriminator.forward(interpolated)
    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=D_interp_logit, inputs=interpolated,
                               grad_outputs=torch.ones_like(D_interp_logit),
                               create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    discriminator_loss_gp = ((gradients_norm - 1) ** 2).mean()    
    
    discriminator_loss = discriminator_loss_imgs + discriminator_loss_decoder + discriminator_loss_fake + discriminator_loss_gp

    if train:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()

    shared = torch.empty(4, dtype=torch.float)
    shared[0] = float(discriminator_loss_imgs)
    shared[1] = float(discriminator_loss_decoder)
    shared[2] = float(discriminator_loss_fake)
    shared[3] = float(discriminator_loss_gp)
    shared = shared.cuda()
    dist.all_reduce(shared, op=dist.ReduceOp.SUM)
    shared = shared.cpu()

    return float(shared[0] / float(world_size)), float(shared[1] / float(world_size)), float(shared[2] / float(world_size)), float(shared[3] / float(world_size))

def test_generation_step(imgs):
    encoder.eval()
    decoder.eval()

    E_mean, E_var = encoder.forward(imgs)
    G_imgs = decoder.forward(E_mean)

    return G_imgs

def save_image_pairs(inp, out, path):
    fig = plt.Figure()
    total = inp.shape[0]
    for i in range(inp.shape[0]):
        img = (np.transpose(inp[i].detach().cpu().numpy(), [1,2,0]) + 1) / 2.0
        ax1 = plt.subplot(total, 2, (i * 2) + 1)
        ax1.imshow(np.squeeze(img))
        outimg = (np.transpose(out[i].detach().cpu().numpy(), [1,2,0]) + 1) / 2.0
        ax2 = plt.subplot(total, 2, (i * 2) + 2)
        ax2.imshow(np.squeeze(outimg))
    plt.savefig(path)

print("Launching!!")
sys.stdout.flush()

for epoch in range(num_epochs):
    train_loader.sampler.set_epoch(epoch)
    encoder_loss = (0.0, 0.0)
    decoder_loss = (0.0, 0.0, 0.0)
    discriminator_loss = (0.0, 0.0, 0.0, 0.0)
    n = 0
    for idx, data in enumerate(train_loader, 0):
        encoder.train()
        decoder.train()
        discriminator.train()
        imgs = data.to(device)
        batch_size = imgs.shape[0]

        current_sum_function = lambda x: ((x[0] * n) + x[1]) / (n + 1)
        discriminator_loss = tuple(map(current_sum_function, zip(discriminator_loss, discriminator_train_step(imgs, batch_size, True))))
        encoder_loss = tuple(map(current_sum_function, zip(encoder_loss, encoder_train_step(imgs, batch_size))))
        decoder_loss = tuple(map(current_sum_function, zip(decoder_loss, decoder_train_step(imgs, batch_size))))

        if (idx % 20 == 0):
            print(idx, encoder_loss, decoder_loss, discriminator_loss)
            sys.stdout.flush()
        n = n + 1
    print('Epoch {}: E: {}, G: {}, D: {}'.format(epoch, encoder_loss, decoder_loss, discriminator_loss))
    if (rank == 0):
        with torch.no_grad():
            for data in val_loader:
                imgs = data
                imgs = imgs.to(device)
                out = test_generation_step(imgs)
                save_image_pairs(imgs, out, "current_pictures/" + save_prefix + "testing.png")
                break
        torch.save(encoder.module.state_dict(), save_prefix+"_encoder.model")
        torch.save(encoder_optimizer.state_dict(), save_prefix+"_encoder.optimizer")
        torch.save(decoder.module.state_dict(), save_prefix+"_decoder.model")
        torch.save(decoder_optimizer.state_dict(), save_prefix+"_decoder.optimizer")
        torch.save(discriminator.module.state_dict(), save_prefix+"_discriminator.model")
        torch.save(discriminator_optimizer.state_dict(), save_prefix+"_discriminator.optimizer")
