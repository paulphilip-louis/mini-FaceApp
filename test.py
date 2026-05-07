import os
import torch
from src.utils import utils
from src.evaluation import evaluation
import argparse

PLOT_PATH = "results"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(model_name, idx=None):
    if not os.path.exists(os.path.join(PLOT_PATH, model_name)):
        os.makedirs(os.path.join(PLOT_PATH, model_name))
    print("Loading model...")
    model = utils.load_model(model_name)
    print("Model loaded")
    print("Loading dataset...")
    dataset = utils.load_CelebA(split='test')
    print("Dataset loaded")
    print("Testing model...")
    if idx:
        utils.plot_reconstruction(model_name, idx, dataset, model, device)
    else:
        evaluation.generate_img(model, device)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a VAE model on CelebA dataset')
    parser.add_argument('--model-name', type=str, required=True, help='Name of the model to test')
    parser.add_argument('--idx', type=int, default=None, help='Index of the image to test')
    args = parser.parse_args()
    main(args.model_name, args.idx)