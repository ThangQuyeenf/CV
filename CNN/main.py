import sys
import numpy as np
import tensorflow as tf
from mnist import MNIST
from cnn import cnn_model




if __name__ == "__main__":
  mndata = MNIST('./data/')
  mndata.load_training()
  train_data = np.asarray(mndata.train_images).reshape(-1, 28, 28, 1)/255.0
  train_labels = np.asarray(mndata.train_labels.tolist())

  mndata.load_testing()
  test_data = np.asarray(mndata.test_images).reshape(-1, 28, 28, 1)/255.0
  test_labels = np.asarray(mndata.test_labels.tolist())
  
  print("Training data shape: ", train_data.shape)
  print("Training labels shape: ", train_labels.shape)
  print("Testing data shape: ", test_data.shape)
  print("Testing labels shape: ", test_labels.shape)

  model = cnn_model()
  # Compile the model
  model.compile(optimizer='sgd',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  # Train the model
  model.fit(train_data, train_labels, epochs=5, batch_size=32)

  # Evaluate the model
  test_loss, test_acc = model.evaluate(test_data, test_labels)
  print('Test accuracy:', test_acc)
 