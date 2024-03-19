import numpy as np
import tensorflow as tf
from resnet_tf import ResNet18TF
from keras.datasets import mnist
from keras.optimizers import Adam
from residual_tf import ResidualBlockTF
  
residual_blocks = [
    # Two start conv mapping
    ResidualBlockTF(64, strides=2),
    ResidualBlockTF(64, strides=2),
    # Next three [conv mapping + identity mapping]
    ResidualBlockTF(128, strides=2),
    ResidualBlockTF(128, strides=2),
    ResidualBlockTF(256, strides=2),
    ResidualBlockTF(256, strides=2),
    ResidualBlockTF(512, strides=2),
    ResidualBlockTF(512, strides=2)
]
if __name__ == '__main__':
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  X_train = X_train/255.0
  X_test = X_test/255.0
  X_train = np.reshape(X_train, (-1, 28, 28, 1))
  X_test = np.reshape(X_test, (-1, 28, 28, 1))
  # Convert data type bo be adaptable to tensorflow computation engine
  y_train = y_train.astype(np.int32)
  y_test = y_test.astype(np.int32)
  print(X_test.shape, X_train.shape)
  opt = Adam(learning_rate=0.001, beta_1= 0.9, beta_2=0.9)

  tfmodel = ResNet18TF(residual_blocks, output_shape=10)
  tfmodel.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  tfmodel.fit(X_train, y_train,
              validation_data = (X_test, y_test), 
              epochs=5, batch_size=32)