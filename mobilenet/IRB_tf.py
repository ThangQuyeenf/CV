import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, DepthwiseConv2D, Add

def inverted_linear_residual_block(x, expand=64, squeeze=16):
  """
  expand: num channels of intermediate layer
  squeeze: num channels of bottleneck layer (input and output)
  """

  # Depthwise convolution
  m = Conv2D(expand, kernel_size=(1,1), padding='SAME', activation='relu')(x)
  m = DepthwiseConv2D(kernel_size=(3,3), padding='SAME', activation='relu')(m)

  # Pointwise convolution
  m = Conv2D(squeeze, kernel_size=(1,1), padding='SAME', activation='relu')(m)

  # Add
  opt = Add()([m, x])

  return opt


if __name__ == '__main__':
  x = np.random.randn(10, 64, 64, 16) # Batch_size, C, W, H
  x = x.astype('float32')
  y = inverted_linear_residual_block(x, expand=64, squeeze=16)
  print(y.shape)