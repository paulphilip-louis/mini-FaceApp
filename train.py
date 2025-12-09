from src.models.engine import VAE
from src.training.training import train_vae
import src.utils.utils as utils
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import torch
import os
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = 'models'

def main(model_name, epochs, batch_size=64, learning_rate=1e-3, beta=1e-4):
    print("Loading dataset...")
    dataset = utils.load_CelebA(split='train')
    print("Dataset loaded")
    print("Creating dataloader...")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Dataloader created")
    print("Creating VAE model...")
    vae_model = VAE(input_shape=(3,128, 128), channels_enc=[16, 32, 64], channels_dec=[32, 16, 3], kernel_enc=[3, 3, 3], kernel_dec=[3, 3, 3], stride_enc=(2, 2), scale_factor=2, latent_dim=200)
    print("VAE model created")
    print("Moving model to device...")
    vae_model.to(device)
    print("Model moved to device")
    criterion = utils.ELBO(beta=beta)
    optimizer = Adam(vae_model.parameters(), lr=learning_rate)
    print("Training model...")
    train_vae(vae_model, dataloader, epochs, criterion, optimizer, device)

    print("Saving model...")
    model_path = os.path.join(MODEL_PATH, model_name)
    utils.save_model(vae_model, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VAE model on CelebA dataset')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--model-name', type=str, default=None, help='Name to save the model (optional)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate for optimizer (default: 1e-3)')
    parser.add_argument('--beta', type=float, default=1e-2, help='Beta parameter for ELBO loss (default: 1e-4)')
    
    args = parser.parse_args()
    
    main(
        epochs=args.epochs,
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta
    )

