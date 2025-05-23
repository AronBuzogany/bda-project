# Autoencoder and VAE Comparison on MNIST

This project compares various dimensionality reduction and generative techniques using the MNIST dataset, including:

- **Autoencoders**
- **Variational Autoencoders (VAE)**
- **Adversarial Autoencoders (AAE)**
- **PCA, LDA, t-SNE, Isomap**

## Contents

- `Autoencoder.ipynb`: Training and evaluating a simple autoencoder, adversarial autoencoder, and classical methods (PCA, LDA, t-SNE, Isomap) on MNIST.
- `VAE.ipynb`: Implementation of a convolutional variational autoencoder (CVAE) with training and sampling functionality.

## Features

### Autoencoder

- Compress and reconstruct MNIST digits
- Evaluate reconstruction error and compression ratio

### Adversarial Autoencoder (AAE)

- Combines an autoencoder with a discriminator to regularize the latent space
- Trained using two optimizers (one for AE, one for discriminator)

### Variational Autoencoder (CVAE)

- Convolutional encoder/decoder
- Samples from a learned latent distribution using reparameterization
- Includes ELBO-based loss function

### Classical Dimensionality Reduction

- PCA
- LDA (no reconstruction)
- t-SNE and Isomap (visualization only)

## Requirements

- Python
- TensorFlow 2.x
- NumPy, Matplotlib, Pandas, Scikit-learn

Install with:

```bash
pip install tensorflow numpy matplotlib pandas scikit-learn
```

## Usage

Run the notebooks:

1. `Autoencoder.ipynb`: Run to train autoencoders and compare with traditional techniques
2. `VAE.ipynb`: Run to train a convolutional variational autoencoder and visualize samples

## Dataset

- MNIST is loaded directly from `tensorflow.keras.datasets`.
