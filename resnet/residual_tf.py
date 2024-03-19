import tensorflow as tf
from keras import Layer, layers, Sequential

class ResidualBlockTF(Layer):
  def __init__(self, filters, strides=1):
    super(ResidualBlockTF, self).__init__()
    self.conv1 = layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False)
    self.bn1 = layers.BatchNormalization()
    self.relu = layers.ReLU()
    self.conv2 = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)
    self.bn2 = layers.BatchNormalization()
    if strides != 1:
      self.downsample = Sequential()

      self.downsample.add(layers.Conv2D(filters, 1, strides=strides, padding='same', use_bias=False))
      self.downsample.add(layers.BatchNormalization())
    else:
      self.downsample = lambda x: x
    
  def call(self, X):
    residual = self.downsample(X)
    X = self.conv1(X)
    X = self.bn1(X)
    X = self.relu(X)
    X = self.conv2(X)
    X = self.bn2(X)
    
    output = self.relu(layers.Add()([residual, X]))
    return output

if __name__ == '__main__':
  X = tf.random.uniform((4, 28, 28, 1)) # shape=(batch_size, width, height, channels)
  X = ResidualBlockTF(filters=64, strides=2)(X)
  print(X.shape)