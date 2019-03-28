import numpy as np
import matplotlib.pyplot as plt
import chanflow as cf

# Global variables.
log_period_samples = 20000
batch_size = 100

from tensorflow.examples.tutorials.mnist import input_data
def get_data():
  return input_data.read_data_sets("MNIST_data/", one_hot=True)

experiments_task2 = []
settings = [(15, 0.0001), (15, 0.005), (15, 0.1)]

# Train Model 2 with the different hyper-parameter settings.
for (num_epochs, learning_rate) in settings:

  # Reset graph, recreate placeholders and dataset.
  mnist = get_data()
  eval_mnist = get_data()

  #####################################################
  # Define model, loss, update and evaluation metric. #
  #####################################################
  
  # First dense layer
  dense1 = cf.layers.dense(layer_sizes=[784, 32], init_weight='xavier', init_bias='zeros')
  
  # Relu activation
  rel = cf.layers.relu()
  
  # Second dense layer
  dense2 = cf.layers.dense(layer_sizes=[32, 10], init_weight='xavier', init_bias='zeros')

  model2 = [dense1, rel, dense2]
  loss_f = cf.utils.cross_entropy_loss
  nn = cf.nn(model2, loss_f)

  # Train.
  i, train_accuracy, test_accuracy = 0, [], []
  log_period_updates = int(log_period_samples / batch_size)
  while mnist.train.epochs_completed < num_epochs:

    # Update.
    i += 1
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)

    #################
    # Training step #
    #################
    loss = nn.train_step(batch_xs, batch_ys, learning_rate)    

    # Periodically evaluate.
    if i % log_period_updates == 0:
      
      # Get a fifth of a shuffled indices of the images
      indices = np.arange(mnist.train.images.shape[0])
      np.random.shuffle(indices)     
      indices_fifth = indices[:int(len(indices) * 0.2)]

      #####################################
      # Compute and store train accuracy. #
      #####################################
      
      accuracy_train = cf.utils.accuracy(
          mnist.train.labels[:int(len(indices) * 0.2)],
          nn.predict(mnist.train.images[:int(len(indices) * 0.2)])
      )

      train_accuracy.append(accuracy_train)

      #####################################
      # Compute and store test accuracy.  #
      #####################################
      accuracy_test = cf.utils.accuracy(
              mnist.test.labels,
              nn.predict(mnist.test.images)
          )

      test_accuracy.append(accuracy_test)
      
      print('accuracy_train',accuracy_train, 'accuracy_test', accuracy_test)

  experiments_task2.append(
      ((num_epochs, learning_rate), train_accuracy, test_accuracy))