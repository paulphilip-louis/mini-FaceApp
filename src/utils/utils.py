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