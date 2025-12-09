import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms.functional import to_pil_image
from torchvision.datasets import CelebA


def load_CelebA(split='train', root="data"):
    """
    Load the CelebA dataset.
    Parameters :
        - split : 'train' or 'test'
        - root (str) : path to the dataset if already downloaded, else the path to the dataset will be created
    Returns :
        - data (torch.utils.data.Dataset) : CelebA dataset 
    """
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
    """
    Plot an image from a torch tensor X (values between 0 and 1)
    """
    img = to_pil_image(X)
    plt.imshow(img)
    return(img)

def plot_reconstruction(model_name, idx, dataset, vae, device, seed=42):
    """
    Plot the image dataset[idx] and the reconstruction of the image by the VAE
    """
    vae = vae.to(device)
    vae.eval()
    face = dataset[idx]
    X_true = face[0].unsqueeze(0).to(device)
    X_reconst, _, _ = vae(X_true)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(X_true.squeeze(0).cpu().numpy().transpose((1,2,0)))
    axs[0].set_title('Original')
    axs[1].imshow(X_reconst.squeeze(0).cpu().detach().numpy().transpose((1,2,0)))
    axs[1].set_title('Reconstruction')
    plt.savefig(f"results/{model_name}/reconstruction_{idx}.png")
    plt.show()

    

def generate_img(vae, device, seed=42, plot=True):
    """
    Generate a random image from the VAE decoder
    Parameters :
        - vae (VAE) : VAE model
        - device (torch.device) : device to use
        - seed (int) : seed for the random number generator
        - plot (bool) : whether to plot the generated image
    Returns :
        - gen (torch.Tensor) : generated image
    """
    torch.manual_seed(seed)
    vae.to(device)
    vae.eval()
    noise = torch.randn(1, vae.latent_dim).to(device)
    gen = vae.decoder(noise).squeeze(0)
    if plot:
        plot_image(gen)

def attribute_vector(vae, data, attribute_idx, min, device):
    """
    Compute the attribute vector for a given attribute index
    Parameters :
        - vae (VAE) : VAE model
        - data (torch.utils.data.Dataset) : CelebA dataset
        - attribute_idx (int) : index of the attribute to compute the attribute vector for
        - min (int) : minimum number of images to generate for the attribute vector
        - device (torch.device) : device to use
    Returns :
        - attr_vector (torch.Tensor) : attribute vector
    """
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

def ELBO(beta):
    """
    Return the ELBO loss function for a given beta
    Parameters :
        - beta (float) : beta parameter for the ELBO loss
    Returns :
        - criterion_beta (function) : ELBO loss function
    """
    def criterion_beta(y1,y2, mu, logvar, beta=beta):
        kl_div = -1/2 * torch.mean(1+logvar-mu.pow(2)-logvar.exp())
        bce_loss = nn.MSELoss()
        bce = bce_loss(y1,y2)
        return bce, beta*kl_div
    return criterion_beta

def save_model(model, path):
    """
    Save the model to a given path
    Parameters :
        - model (VAE) : VAE model
        - path (str) : path to save the model
    """
    torch.save(model.state_dict(), path)

def load_model(path):
    """
    Load the model from a given path
    Parameters :
        - path (str) : path to load the model
    Returns :
        - model (VAE) : VAE model
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file {path} not found")
    model = torch.load(path)
    return model