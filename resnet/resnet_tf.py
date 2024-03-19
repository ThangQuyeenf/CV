import tensorflow as tf
from keras import Model, layers
from residual_tf import ResidualBlockTF

class ResNet18TF(Model):
  def __init__(self, residual_blocks, output_shape):
    super(ResNet18TF, self).__init__()
    self.conv1 = layers.Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu', input_shape=(224,224,3))
    self.batch_norm = layers.BatchNormalization()
    self.maxpool = layers.MaxPool2D(pool_size=(3,3), strides=2, padding='same')
    self.residual_blocks = residual_blocks
    self.global_avg_pool = layers.GlobalAveragePooling2D()
    self.dense = layers.Dense(units=output_shape)
  
  def call(self, x):
    x = self.conv1(x)
    x = self.batch_norm(x)
    x = self.maxpool(x)
    for residual_block in self.residual_blocks:
      x = residual_block(x)
    x = self.global_avg_pool(x)
    x = self.dense(x)
    return x
  
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


if __name__ == "__main__":
  tfmodel = ResNet18TF(residual_blocks, output_shape=10)
  tfmodel.build(input_shape=(None, 28, 28, 1))
  tfmodel.summary()