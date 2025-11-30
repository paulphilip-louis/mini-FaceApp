import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms.functional import to_pil_image
from torchvision.datasets import CelebA


def load_CelebA(split='train', root="data"):
    if split != 'train' and split != 'test':
        raise ValueError("split must be either train or test !")
    data = CelebA(
        root=root,
        split=split,
        download=True,
        transform=Compose([
            Resize((128, 128)),
            ToTensor()])
    ) 
    return data


def plot_image(X):
    img = to_pil_image(X)
    plt.imshow(img)
    return(img)

def generate_img(vae, device, seed=42, plot=True):
    torch.manual_seed(seed)
    vae.to(device)
    vae.eval()
    noise = torch.randn(1, vae.latent_dim).to(device)
    gen = vae.decoder(noise).squeeze(0)
    if plot:
        plot_image(gen)

def attribute_vector(vae, data, attribute_idx, min, device):
    X_true = []
    X_false = []
    vae.to(device)
    vae.eval()
    idx=0
    while len(X_true)<min and idx<len(data):
        face = data[idx]
        if face[1][attribute_idx].item()==1:
            X_true.append(face[0].unsqueeze(0)) 
        else:
            X_false.append(face[0].unsqueeze(0))
        idx+=1
    X_true = torch.concatenate(X_true, axis=0).to(device)
    X_false = torch.concatenate(X_false, axis=0).to(device)

    true_enc, _, _ = vae.encoder(X_true)
    false_enc, _, _ = vae.encoder(X_false)

    attr_vector = torch.mean(true_enc, axis=0) - torch.mean(false_enc, axis=0)

    return attr_vector

def criterion(beta):
    def criterion_beta(y1,y2, mu, logvar, beta=1e-2):
        kl_div = -1/2 * torch.mean(1+logvar-mu.pow(2)-logvar.exp())
        bce_loss = nn.BCELoss()
        bce = bce_loss(y1,y2)
        return bce, beta*kl_div
    return criterion_beta