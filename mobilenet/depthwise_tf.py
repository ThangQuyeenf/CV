import tensorflow as tf
import numpy as np
from keras.layers import DepthwiseConv2D, Conv2D



def depthwise_separable_conv_tf(x):
  dw2d = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1)

  pw2d = Conv2D(filters=50, kernel_size=1, strides=1)

  y = dw2d(x)
  y = pw2d(y)

  return y

if __name__ == '__main__':
  x = np.random.randn(10, 15, 15, 20)
  x = x.astype(np.float32)
  y = depthwise_separable_conv_tf(x)
  print(y.shape)