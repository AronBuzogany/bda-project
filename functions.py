import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE, Isomap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from time import time

# Adversarial autoencoder
class AdversarialAutoencoder(tf.keras.Model):
    def __init__(self, latent_dim, shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(np.prod(shape), activation='sigmoid'),
            layers.Reshape(shape)
        ])
        self.discriminator = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

    def compile(self, ae_optimizer, d_optimizer, loss_fn):
        super().compile()
        self.ae_optimizer = ae_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
        self.bce = tf.keras.losses.BinaryCrossentropy()

    def train_step(self, data):
        x = data
        batch_size = tf.shape(x)[0]

        # Encode
        with tf.GradientTape(persistent=True) as tape:
            z = self.encoder(x)
            x_reconstructed = self.decoder(z)

            real = tf.random.normal(shape=(batch_size, self.latent_dim))
            fake = z

            real_output = self.discriminator(real)
            fake_output = self.discriminator(fake)

            ae_loss = self.loss_fn(x, x_reconstructed)
            d_loss = self.bce(tf.ones_like(real_output), real_output) + \
                     self.bce(tf.zeros_like(fake_output), fake_output)

        grads_ae = tape.gradient(ae_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        grads_d = tape.gradient(d_loss, self.discriminator.trainable_variables)

        self.ae_optimizer.apply_gradients(zip(grads_ae, self.encoder.trainable_variables + self.decoder.trainable_variables))
        self.d_optimizer.apply_gradients(zip(grads_d, self.discriminator.trainable_variables))

        return {"ae_loss": ae_loss, "d_loss": d_loss}

# Comparison setup
def evaluate_methods(x_train, y_train, x_test, y_test, latent_dim=10):
    results = {}

    # Autoencoder
    ae = train_autoencoder(x_train, x_test, latent_dim)
    encoded_ae = ae.encoder(x_test).numpy()
    decoded_ae = ae.decoder(encoded_ae).numpy()
    results["AE"] = {
        "recon": mean_squared_error(x_test.reshape(len(x_test), -1), decoded_ae.reshape(len(x_test), -1)),
        "comp_ratio": np.prod(x_test.shape[1:]) / latent_dim,
        "time": None
    }

    # PCA
    start = time()
    pca = PCA(n_components=latent_dim).fit(x_train.reshape(len(x_train), -1))
    encoded = pca.transform(x_test.reshape(len(x_test), -1))
    recon = pca.inverse_transform(encoded)
    duration = time() - start
    results["PCA"] = {
        "recon": mean_squared_error(x_test.reshape(len(x_test), -1), recon),
        "comp_ratio": np.prod(x_test.shape[1:]) / latent_dim,
        "time": duration
    }

    # LDA
    start = time()
    lda = LDA(n_components=1).fit(x_train.reshape(len(x_train), -1), y_train)
    encoded = lda.transform(x_test.reshape(len(x_test), -1))
    duration = time() - start
    results["LDA"] = {
        "comp_ratio": np.prod(x_test.shape[1:]) / 1,
        "time": duration,
        "recon": None  # no reconstruction in LDA
    }

    # t-SNE
    start = time()
    tsne = TSNE(n_components=min(3, latent_dim)).fit_transform(x_test[:1000].reshape(1000, -1))
    duration = time() - start
    results["t-SNE"] = {
        "comp_ratio": np.prod(x_test.shape[1:]) / latent_dim,
        "time": duration,
        "recon": None  # no reconstruction in t-SNE
    }

    # Isomap
    start = time()
    isomap = Isomap(n_components=latent_dim).fit(x_test[:1000].reshape(1000, -1))
    encoded = isomap.transform(x_test[:1000].reshape(1000, -1))
    duration = time() - start
    results["Isomap"] = {
        "comp_ratio": np.prod(x_test.shape[1:]) / latent_dim,
        "time": duration,
        "recon": None  # no reconstruction in Isomap
    }

    return results


def load_mnist_data():
  #Loading mnist dataset
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train.astype('float32') / 255.
  x_test = x_test.astype('float32') / 255.
  return x_train, y_train, x_test, y_test

#From tensorflow tutorial
#https://www.tensorflow.org/tutorials/generative/autoencoder
class Autoencoder(Model):
  def __init__(self, latent_dim, shape):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'),
      layers.Reshape(shape)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

def train_autoencoder(x_train, x_test, latent_dim, optimizer='adam', loss=losses.MeanSquaredError()):
  #Trains autoencoder with gven latent dimension
  shape = x_test.shape[1:]
  autoencoder = Autoencoder(latent_dim, shape)
  autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
  autoencoder.fit(x_train, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test))
  return autoencoder

def get_reconstructed_img(autoencoder, x_test):
  #Get encoded and decoded images
  encoded_imgs = autoencoder.encoder(x_test).numpy()
  decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
  return encoded_imgs, decoded_imgs

def plot_reconstruction(x_test, decoded_imgs, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def generate_rectangle(label, size=28):
  rectangle = np.zeros((size, size))
  if label == 0:
    start_x = np.random.randint(0, 22)
    start_y = np.random.randint(0, 17)
    rectangle[start_y:start_y+10, start_x:start_x+5] = 1
  if label == 1:
    start_x = np.random.randint(0, 17)
    start_y = np.random.randint(0, 22)
    rectangle[start_y:start_y+5, start_x:start_x+10] = 1
  return rectangle

def get_rectangle_img(length=70000, seed=1234):
  np.random.seed(seed)
  labels = np.random.randint(0, 2, size=length)
  rectangles = np.array([generate_rectangle(i) for i in labels])
  return rectangles

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits
  
def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)



def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def train_vae(epochs, model, train_dataset, test_dataset, optimizer):
  for epoch in range(1, epochs + 1):
    start_time = time()
    for train_x in train_dataset:
      train_step(model, train_x, optimizer)
    end_time = time()

    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
      loss(compute_loss(model, test_x))
    elbo = -loss.result()
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
          .format(epoch, elbo, end_time - start_time))