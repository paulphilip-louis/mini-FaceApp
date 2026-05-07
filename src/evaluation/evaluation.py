import torch
from src.utils import utils


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
        utils.plot_image(gen)

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