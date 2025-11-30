import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.optim import Adam
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pickle as pkl
from torchvision.transforms.functional import to_pil_image


class Encoder(nn.Module):
  def __init__(self, input_shape, latent_dim, channels, kernels, stride=(2,2)):
    """
    input_shape : (Channels, Height, Width)
    kernel_size : (height, width)
    N convolutional layers -> channels_in, channels_out, kernel_size, stride for each
    Flatten
    2 Linear layers -> mu, logvar
    One Lambda layer
    """
    super().__init__()

    self.n_layers = len(channels)

    strides = [stride]*self.n_layers

    layers = []
    in_chan = input_shape[0]

    for out_chan, k, s in zip(channels, kernels, strides):
      layers.append(nn.Conv2d(in_chan, out_chan, k, s, padding=1))
      layers.append(nn.LeakyReLU())
      in_chan = out_chan

    self.layers = nn.Sequential(*layers)
    self.flatten = nn.Flatten()

    with torch.no_grad():
      dummy = torch.zeros(1, *input_shape)
      dummy_output = self.layers(dummy)
      self.inner_shape = dummy_output.shape[1:]
      self.flat_size = dummy_output.flatten(1).shape[1]

    self.mu = nn.Linear(self.flat_size, latent_dim)
    self.logvar = nn.Linear(self.flat_size, latent_dim)

  def forward(self, x):
    x = self.layers(x)
    x = self.flatten(x)
    mu, logvar = self.mu(x), self.logvar(x)
    output = mu + torch.exp(0.5*logvar)*torch.randn_like(logvar)
    return output, mu, logvar


class Decoder(nn.Module):
  def __init__(self, inner_shape, flat_size, latent_dim, channels, kernels, scale_factor=2, target_shape=None):
    super().__init__()

    self.n_layers = len(channels)
    self.inner_shape = inner_shape
    self.target_shape = target_shape

    scale_factors = [scale_factor]*self.n_layers

    
    """
    for out_chan, k, s in zip(channels, kernels, scale_factors):
      layers.append(nn.ConvTranspose2d(in_chan, out_chan, k, s))
      layers.append(nn.BatchNorm2d(out_chan))
      layers.append(nn.LeakyReLU())
      layers.append(nn.Dropout(p=0.15))
      in_chan = out_chan
    """
    layers = []
    in_chan = inner_shape[0]
    for i, (out_chan, k) in enumerate(zip(channels, kernels)):
      if i>0:
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))

      layers.append(nn.Conv2d(in_chan, out_chan, kernel_size=k, stride=1, padding=1))

      if i < len(channels)-1:
        layers.append(nn.BatchNorm2d(out_chan))
        layers.append(nn.LeakyReLU(0.2))

      in_chan = out_chan
    
    self.fc = nn.Linear(latent_dim, flat_size)#, bias=False)
    self.conv = nn.Sequential(*layers)
    self.final_act = nn.Sigmoid()


  def forward(self, x):
    x = self.fc(x)
    x = x.reshape((x.size(0),)+self.inner_shape)
    x = self.conv(x)
    x = self.final_act(x)
    if self.target_shape is not None:
      x = nn.functional.interpolate(x, size=self.target_shape, mode='bilinear', align_corners=False)
    return x
  

class VAE(nn.Module):
  def __init__(self, input_shape, channels_enc, channels_dec, kernel_enc, kernel_dec, stride_enc=(2,2), scale_factor=2, latent_dim = 200, target_shape=None):
    """
    Initialize an instance of variational auto-encoder.
    Parameters :
      - input_shape : shape of the image (channel, height, width)
      - channels encoder : number of channels between each layer of the encoder
      - channels decoder : same for the decoder
      - kernel_enc : array containing in order the kernel size of each convolutional layer of the encoder
      - kernel_dec : array containing in order the kernel size of each convolutional layer of the decoder
      - stride_enc : array containing in order the stride of each convolutional layer of the encoder
      - stride_dec : array containing in order the stride of each convolutional layer of the decoder
      - latent_dim : dimension of the latent space to which we project
    """
    super().__init__()
    self.latent_dim = latent_dim
    self.encoder = Encoder(input_shape=input_shape, latent_dim=latent_dim, channels=channels_enc, kernels=kernel_enc, stride=stride_enc)
    self.decoder = Decoder(inner_shape=self.encoder.inner_shape, flat_size=self.encoder.flat_size, latent_dim=latent_dim, channels=channels_dec, kernels=kernel_dec, scale_factor=scale_factor, target_shape=target_shape)

  def forward(self, x):
    x, mu, logvar = self.encoder(x)
    x = self.decoder(x)
    return x, mu, logvar

def train_vae(vae, dataloader, nb_epochs, criterion, optimizer, device):
  vae.to(device)
  vae.train()
  for epoch in range(nb_epochs):
    for idx, (X_batch, _) in tqdm(enumerate(dataloader)):
      optimizer.zero_grad()
      X_batch = X_batch.to(device)
      X_reconst, mu, logvar = vae(X_batch)
      bce, kl = criterion(X_reconst, X_batch, mu, logvar)
      loss = bce+kl
      loss.backward()
      optimizer.step()
      if idx%100==0:
        print(f"Epoch {epoch}, Batch {idx} : BCE = {bce}, KL = {kl}")