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