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

- Output : $x = \mu + \exp(\frac{1}{2}logvar).\epsilon$ with $\epsilon \sim \mathcal{N}(0,I)$

**Decoder**
- Successive blocks composed of
  - Upsampling + Convolutional layer
  - BatchNorm + LeakyReLU after
  - For final block : Sigmoid activation
 
**Latent space**
- Configurable dimension

**Loss**\

$\mathcal{L}(\phi, \theta) = - \mathrm{ELBO}(\phi,\theta) = \beta\mathrm{KL}\big(q_{\phi}(z\mid x)\|\|p_{\lambda}(z)\big) - \mathbb{E}\_{q_{\phi}(z\mid x)} \left[\log p_{\theta}(x\mid z)\right] $

The VAE build is thus technically a Beta-VAE. In practice, the loss used is :\
```nn.BCELoss(y1, y2) + beta * (-1/2) * torch.mean(1+logvar-mu.pow(2)-logvar.exp())```

## 3. Architecture
```
mini-FaceApp/
│
├── src/
│   ├── models/          # VAE, encodeur, décodeur
│   ├── training/        # boucle d'entraînement, pertes, gestion checkpoints
│   ├── data/            # dataloaders, préprocessing
│   ├── evaluation/      # génération d’images, interpolation, métriques
│   └── utils/           # fonctions auxiliaires
│
├── experiments/
│   ├── exp_01_baseline.md
│   ├── exp_02_latent_dim.md
│   └── ...
│
├── notebooks/
│   └── demo.ipynb       # démonstrations, visualisations
│
├── docs/
│   └── architecture.md  # schémas, décisions techniques
│
├── results/             # sorties générées (reconstructions, interpolations)
│
├── requirements.txt
├── README.md
└── .gitignore
```
## 4. Installation
First make sure you installed ```uv``` first.

```uv sync```.

## 5. Utilisation

TO DO




