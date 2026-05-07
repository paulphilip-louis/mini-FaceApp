import torch
import torch.nn as nn
from tqdm import tqdm


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


def train_vae(vae, dataloader, nb_epochs, criterion, optimizer, device):
  """
  Train the VAE model
  Parameters :
    - vae (VAE) : VAE model
    - dataloader (torch.utils.data.DataLoader) : dataloader for the training data
    - nb_epochs (int) : number of epochs to train the model
    - criterion (function) : criterion function to use for the training
    - optimizer (torch.optim.Optimizer) : optimizer to use for the training
    - device (torch.device) : device to use for the training
  """
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