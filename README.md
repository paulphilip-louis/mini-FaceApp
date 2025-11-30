# MiniFaceApp

Implementation from scratch of a Variational Auto-Encoder (VAE) in PyTorch trained on the CelebA dataset.

This project provides a modular architecture, training scripts and reproducible evaluation scripts, as well as documentated experiments.

## 1. Goal

Propose a clear and entirely modular implementation of a VAE applied to face reconstruction and generation.

## 2. Architecture

**Encoder**
- Progressive convolutional layers
- BatchNorm layer and LeakyReLU activation after each convolutional layer
- Projection to two values : `mu` and `logvar` for the reparametrization trick
- Output : $x = \mu + logvar.\epsilon$ with $\epsilon \tilde N(0,1)$ 