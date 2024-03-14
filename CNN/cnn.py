import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers, models
# tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model():
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
  model = models.Sequential()
  # Convolutional Layer #1
  model.add(layers.Conv2D(
      filters=32,
      kernel_size=(5, 5),
      padding="same",
      activation=tf.nn.relu,
      input_shape=(28, 28, 1)
  ))
  # Apply formula: N1 = (N+2P-F)/S +1
  # Output tensor shape: N1 = (28-5)/1+1 => shape = [-1, 24, 24, 1]
  # But we at parameter we set padding = 'same' in order to keep output shape unchange to input shape
  # Thus Output shape is [-1, 28, 28, 1]

  # Max pooling layer 1
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
  # Output tensor shape: N2 = (28-2)/2+1 = 14 => shape = [-1, 14, 14, 1]
  
  # Convolutional Layer #2
  model.add(layers.Conv2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        activation=tf.nn.relu
  ))
  # Output tensor shape: N3 = (14-5)/1+1 = 10 => shape = [-1, 10, 10, 1]
  # But padding = 'same' so output shape is [-1, 14, 14, 1]

  # Max pooling layer 2
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
  # Output tensor shape: N2 = (14-2)/2+1 = 14 => shape = [-1, 7, 7, 1]
  
  # Flatten tensor into a batch of vectors
  model.add(layers.Flatten())
  
  # Dense Layer
  model.add(layers.Dense(units=1024, activation=tf.nn.relu))

  model.add(layers.Dropout(rate=0.4))

  # Logits layer
  model.add(layers.Dense(units=10, activation=None))

  return model
  