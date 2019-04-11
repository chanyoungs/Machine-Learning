# Import useful libraries.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Global variables.
log_period_samples = 20000
batch_size = 100

# Import dataset with one-hot encoding of the class labels.
def get_data():
  return input_data.read_data_sets("MNIST_data/", one_hot=True)

# Placeholders to feed train and test data into the graph.
# Since batch dimension is 'None', we can reuse them both for train and eval.
def get_placeholders():
  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])
  return x, y_

num_epochs = 5
learning_rate = 0.001
  
# Reset graph, recreate placeholders and dataset.
tf.reset_default_graph()  # reset the tensorflow graph
x, y_ = get_placeholders()
mnist = get_data()  # use for training.
eval_mnist = get_data()  # use for evaluation.

#####################################################
# Define model, loss, update and evaluation metric. #
#####################################################

# Initialise weights with xavier
initializer = tf.contrib.layers.xavier_initializer()
W1 = tf.Variable(initializer([784, 32]))
W2 = tf.Variable(initializer([32, 10]))
# Biases are usually initialised with zeros
b1 = tf.Variable(tf.zeros([32]))  
b2 = tf.Variable(tf.zeros([10]))  

# Define model
y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
y = tf.matmul(y1, W2) + b2

# Note that although reduce_mean is what would be usally used, reduce_sum has
# been used here. Since the batch size is constant, this is equivalent of
# multiplying the learning rate by the batch size.
cross_entropy = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    )
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# Calculate the accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train.
i, train_accuracy, test_accuracy = 0, [], []
log_period_updates = int(log_period_samples / batch_size)
with tf.train.MonitoredSession() as sess:
  while mnist.train.epochs_completed < num_epochs:
    
    # Update.
    i += 1
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    print(batch_xs.shape, batch_ys.shape)
    
    #################
    # Training step #
    #################      

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    # Periodically evaluate.
    if i % log_period_updates == 0:
      
      #####################################
      # Compute and store train accuracy. #
      #####################################
      # Get a fifth of a shuffled indices of the images
      indices = range(mnist.train.images.shape[0])
      np.random.shuffle(indices)     
      indices_fifth = indices[:int(len(indices) * 0.2)]
      
      # Calculate accuracy
      acc_train = sess.run(
              accuracy,
              feed_dict={
                  x: mnist.train.images[indices_fifth],
                  y_: mnist.train.labels[indices_fifth]
                  }
                  )
      
      acc_test = sess.run(
        accuracy,
        feed_dict={
          x: mnist.test.images,
          y_: mnist.test.labels
          }
          )
      
      print(acc_train, acc_test)

      train_accuracy.append(acc_train)
      test_accuracy.append(acc_test)