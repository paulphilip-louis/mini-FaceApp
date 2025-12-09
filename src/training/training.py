import torch
from tqdm import tqdm


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