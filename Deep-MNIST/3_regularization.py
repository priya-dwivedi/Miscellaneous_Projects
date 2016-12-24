# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 15:36:12 2016

@author: priyankadwivedi
"""


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

import os
os.chdir(r"/Users/priyankadwivedi")

## First reload the data from before
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
      
## Problem 4: Lets play around and visualize
import matplotlib
import matplotlib.pyplot as plt
n=1000
image_array = (train_dataset[n])
image_array.shape

plt.imshow(image_array, cmap='Greys', interpolation='None')
print(train_labels[n])
print(train_labels.shape)
## reformat to a better shape


image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)



def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
"""
## Problem 1: Introduce and tune a L2 regularization
## Start with the relu model from fully connected 

batch_size = 128
num_nodes = 1024

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights_1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_nodes]))
  biases_1 = tf.Variable(tf.zeros([num_nodes]))
  weights_2 = tf.Variable(
        tf.truncated_normal([num_nodes, num_labels]))
  biases_2 = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
  logits = tf.matmul(hidden1, weights_2) + biases_2
  loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))


  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(biases_1) +
                  tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(biases_2))
  # Add the regularization term to the loss.
  factor = 5e-4
  loss += factor * regularizers
      
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1), weights_2) + biases_2)
  test_prediction = tf.nn.softmax(
    tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1), weights_2) + biases_2)

num_steps = 5001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

#Minibatch accuracy: 91.4%
#Validation accuracy: 86.1%
#Test accuracy: 92.9%

# Problem 2: Extreme case of overfitting. Restrict your training data to just a few batches
batch_size = 20
num_nodes = 1024

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights_1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_nodes]))
  biases_1 = tf.Variable(tf.zeros([num_nodes]))
  weights_2 = tf.Variable(
        tf.truncated_normal([num_nodes, num_labels]))
  biases_2 = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
  logits = tf.matmul(hidden1, weights_2) + biases_2
  loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))


  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(biases_1) +
                  tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(biases_2))
  # Add the regularization term to the loss.
  factor = 5e-4
  loss += factor * regularizers
      
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1), weights_2) + biases_2)
  test_prediction = tf.nn.softmax(
    tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1), weights_2) + biases_2)

num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

#Minibatch accuracy: 85%
#Validation accuracy: 75.4%%
#Test accuracy: 83.6%
"""
"""
#Problem 4: Introduce dropout to the hidden layer. First with noregularization and then
# with regularization


batch_size = 256
num_nodes = 1024

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights_1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_nodes]))
  biases_1 = tf.Variable(tf.zeros([num_nodes]))
  weights_2 = tf.Variable(
        tf.truncated_normal([num_nodes, num_labels]))
  biases_2 = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
  
  #add dropout on hidden layer
  #we pick up the probabylity of switching off the activation
  #and perform the switch off of the activations
  
  keep_prob = tf.placeholder(tf.float32)
  keep_prob = 0.50
  hidden_layer_drop = tf.nn.dropout(hidden1, keep_prob) 
  
  #final layer takes in the hidden layer drop
  logits = tf.matmul(hidden_layer_drop, weights_2) + biases_2
  loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))


  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(biases_1) +
                  tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(biases_2))
  # Add the regularization term to the loss.
  factor = 5e-4
  loss += factor * regularizers
   
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1), weights_2) + biases_2)
  test_prediction = tf.nn.softmax(
    tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1), weights_2) + biases_2)

num_steps = 5001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

#Dropout but no L2 regularization. Keep only 50% of records.Batch size: 128
#Minibatch accuracy: 77.3%
#Validation accuracy: 78.6%
#Test accuracy: 85.9%

#Minibatch accuracy: 85.2%
#Validation accuracy: 85.0%
#Test accuracy: 91.6%
#L2 regularization increases accuray over dropout.But dropout has very low accuracy. So increasing batch size

#Minibatch accuracy: 82.0%
#Validation accuracy: 85.8%
#Test accuracy: 92.8%
"""
"""
#Problem 5: Multi layer model with L2 regularization

batch_size = 128
n_hidden_1 = 1024 # 1st layer number of features
n_hidden_2 = 1024 # 2nd layer number of features

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Create model
  def multilayer_perceptron(x, weights, biases):
      # Hidden layer with Sigmoid activation
      layer_1 = tf.matmul(x, weights['h1'])+ biases['b1']
      layer_1 = tf.nn.sigmoid(layer_1)
      
      # Hidden layer with RELU activation
      layer_2 = tf.matmul(layer_1, weights['h2'])+ biases['b2']
      layer_2 = tf.nn.sigmoid(layer_2)
      n
      # Output layer with linear activation
      out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
      return out_layer
    
  # Variables.
  # Store layers weight & bias
  weights = {
      'h1': tf.Variable(tf.truncated_normal([image_size * image_size, n_hidden_1])),
      'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
      'out': tf.Variable(tf.truncated_normal([n_hidden_2, num_labels]))
    }
  biases = {
      'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
      'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
      'out': tf.Variable(tf.truncated_normal([num_labels]))
    }

  #Construct model
  logits = multilayer_perceptron(tf_train_dataset, weights, biases)
  
  #Compute the loss function 
  loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

 
  # L2 regularization for the fully connected parameters.
  regularizers =  (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(biases['b1']) +
                  tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(biases['b2']) +
                  tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(biases['out']))
                  
  # Add the regularization term to the loss.
  factor = 5e-4
  loss += factor * regularizers
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(multilayer_perceptron(tf_valid_dataset, weights, biases))
  test_prediction = tf.nn.softmax(multilayer_perceptron(tf_test_dataset, weights, biases))

num_steps = 5001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


# 1 hidden layer and L2 regularization
#Minibatch accuracy: 91.4%
#Validation accuracy: 86.1%
#Test accuracy: 92.9%

# 2 hidden layers and L2 regularization and sigmoid function and 512 nodes
#Minibatch accuracy: 89.1%
#Validation accuracy: 85.0%
#Test accuracy: 91.9%

# 2 hidden layers and L2 regularization and sigmoid function and 1024 nodes. Much slower with no benefit!
#Minibatch accuracy: 89.1%
#Validation accuracy: 84.3%
#Test accuracy: 91.3%
"""
"""
#Problem 5.2: Multi layer model with L2 regularization and learning rate 

batch_size = 128
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Create model
  def multilayer_perceptron(x, weights, biases):
      # Hidden layer with Sigmoid activation
      layer_1 = tf.matmul(x, weights['h1'])+ biases['b1']
      layer_1 = tf.nn.relu(layer_1)
      
      # Hidden layer with Sigmoid activation
      layer_2 = tf.matmul(layer_1, weights['h2'])+ biases['b2']
      layer_2 = tf.nn.sigmoid(layer_2)
      
      # Output layer with linear activation
      out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
      return out_layer
    
  # Variables.
  # Store layers weight & bias
  weights = {
      'h1': tf.Variable(tf.truncated_normal([image_size * image_size, n_hidden_1])),
      'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
      'out': tf.Variable(tf.truncated_normal([n_hidden_2, num_labels]))
    }
  biases = {
      'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
      'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
      'out': tf.Variable(tf.truncated_normal([num_labels]))
    }

  #Construct model
  logits = multilayer_perceptron(tf_train_dataset, weights, biases)
  
  #Compute the loss function 
  loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

 
  # L2 regularization for the fully connected parameters.
  regularizers =  (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(biases['b1']) +
                  tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(biases['b2']) +
                  tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(biases['out']))
                  
  # Add the regularization term to the loss.
  factor = 5e-4
  loss += factor * regularizers
  
  # Define learning rate 
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.5
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.90, staircase=True)
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(multilayer_perceptron(tf_valid_dataset, weights, biases))
  test_prediction = tf.nn.softmax(multilayer_perceptron(tf_test_dataset, weights, biases))

num_steps = 5001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


# Multi layer and L2 regularization and learning rate decay rate of 0.96
#Minibatch accuracy: 87.5%
#Validation accuracy: 85.2%
#Test accuracy: 92.0%

# Multi layer and L2 regularization and learning rate decay rate of 0.50
#Minibatch accuracy: 78.9%
#Validation accuracy: 80.5%
#Test accuracy: 87.5%

# Multi layer and L2 regularization and learning rate decay rate of 0.90. 
# 1 hidden layer has relu and the other sigmoid 
#Minibatch accuracy: 78.9%
#Validation accuracy: 80.5%
#Test accuracy: 87.5%

"""
## Brute force model -  -4X the no of steps 

batch_size = 128
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Create model
  def multilayer_perceptron(x, weights, biases):
      # Hidden layer with Sigmoid activation
      layer_1 = tf.matmul(x, weights['h1'])+ biases['b1']
      layer_1 = tf.nn.relu(layer_1)
      
      # Hidden layer with Sigmoid activation
      layer_2 = tf.matmul(layer_1, weights['h2'])+ biases['b2']
      layer_2 = tf.nn.sigmoid(layer_2)
      
      # Output layer with linear activation
      out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
      return out_layer
    
  # Variables.
  # Store layers weight & bias
  weights = {
      'h1': tf.Variable(tf.truncated_normal([image_size * image_size, n_hidden_1])),
      'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
      'out': tf.Variable(tf.truncated_normal([n_hidden_2, num_labels]))
    }
  biases = {
      'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
      'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
      'out': tf.Variable(tf.truncated_normal([num_labels]))
    }

  #Construct model
  logits = multilayer_perceptron(tf_train_dataset, weights, biases)
  
  #Compute the loss function 
  loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

 
  # L2 regularization for the fully connected parameters.
  regularizers =  (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(biases['b1']) +
                  tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(biases['b2']) +
                  tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(biases['out']))
                  
  # Add the regularization term to the loss.
  factor = 5e-4
  loss += factor * regularizers
  
  # Define learning rate 
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.5
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           3500, 0.86, staircase=True)
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(multilayer_perceptron(tf_valid_dataset, weights, biases))
  test_prediction = tf.nn.softmax(multilayer_perceptron(tf_test_dataset, weights, biases))

num_steps = 20001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 1000 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

#Yay!!!
#Minibatch accuracy: 85.9%
#Validation accuracy: 89.1%
#Test accuracy: 94.9%
