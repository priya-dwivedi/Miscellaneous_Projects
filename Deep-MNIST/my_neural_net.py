
# coding: utf-8

# In[16]:
# Implement neural network on nmist now

import numpy as numpy
import scipy.special

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
    # Initialize the no. of nodes
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # Learning rate
        self.lr = learningrate
        
        #weights inside the arrays are w_i_j, where link is from node i to node j in the next layer #w11 w21 #w12 w22 etc
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)) 
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        #activation function is the sigmoid function 
        self.activation_function = lambda x: scipy.special.expit(x)

        pass
    
    # Train your neural network
    def train(self, input_list, target_list):
        # convert input list to 2d arrat
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T
        
        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        #calculate signals into final output layer 
        final_inputs = numpy.dot(self.who, hidden_outputs) 
        #calculate the signals emerging from final output layer 
        final_outputs = self.activation_function(final_inputs)
        
        # Calculate error
        output_errors = targets - final_outputs
        #hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        #update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors* final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        #update the weights for the links between the input and hidden layers
        self.wih+= self.lr * numpy.dot((hidden_errors* hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass
    
    #Query your neural network
    def query(self, input_list):
        # convert input list to 2d arrat
        inputs = numpy.array(input_list, ndmin=2).T
        
        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        #calculate signals into final output layer 
        final_inputs = numpy.dot(self.who, hidden_outputs) 
        #calculate the signals emerging from final output layer 
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.1

#Create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


# load nmist data. Load trial version first
# Current inputs range b/w -0.5 and 0.5. Convert to 0.01 to 0.99
#num_samples = 200000
num_samples = 1000
n_classes = 10

(samples, width, height) = train_dataset.shape
X = np.reshape(train_dataset,(samples, width*height))[0:num_samples]
y = train_labels[0:num_samples]

y


inputs = ((X[0]+0.5)*0.98)+0.01
inputs
targets = numpy.zeros(output_nodes) + 0.01
targets[y[0]] = 0.99 
targets
  
#len(X)
for record in range(len(X)-1):
    inputs = ((X[record]+0.5)*0.98) +0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[y[record]] = 0.99 
    n.train(inputs,targets)
    pass


# After training, check performance on test
#Basic check first

(samples, width, height) = test_dataset.shape
X_test = np.reshape(test_dataset, (samples, width*height))
y_test = test_labels

X_test.shape
inputs = ((X_test[0]+0.5)*0.98) +0.01
outputs = n.query(inputs)
outputs
label = numpy.argmax(outputs)
correct_label = y_test[0]

print(label)
print(correct_label)

scorecard = []
for record in range(len(X_test)):
   inputs = ((X_test[record]+0.5)*0.98) +0.01
   outputs = n.query(inputs)
   label = numpy.argmax(outputs)
   correct_label = y_test[record]
   if (label== correct_label):
        scorecard.append(1)
   else:
        scorecard.append(0)
        
scorecard_array = numpy.asarray(scorecard) 
print ("performance =", (scorecard_array.sum()*1.00/ scorecard_array.size*1.00))

## Run above neural network on all the data and then check performance
# Current inputs range b/w -0.5 and 0.5. Convert to 0.01 to 0.99
# Load all data sets using pickle

import os
os.chdir(r"/Users/priyankadwivedi")
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

num_samples = 200000
#num_samples = 1000
n_classes = 10

(samples, width, height) = train_dataset.shape
X = np.reshape(train_dataset,(samples, width*height))[0:num_samples]
y = train_labels[0:num_samples]

# Train on all data. Run using epoch
epochs = 5

for e in range(epochs):
    for record in range(len(X)-1):
        inputs = ((X[record]+0.5)*0.98) +0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[y[record]] = 0.99 
        n.train(inputs,targets)
        pass
    pass

# Test again on all test data
(samples, width, height) = test_dataset.shape
X_test = np.reshape(test_dataset, (samples, width*height))
y_test = test_labels

scorecard = []
for record in range(len(X_test)):
   inputs = ((X_test[record]+0.5)*0.98) +0.01
   outputs = n.query(inputs)
   label = numpy.argmax(outputs)
   correct_label = y_test[record]
   if (label== correct_label):
        scorecard.append(1)
   else:
        scorecard.append(0)
        
scorecard_array = numpy.asarray(scorecard) 
print ("performance =", (scorecard_array.sum()*1.00/ scorecard_array.size*1.00))


# Test again on all valid data
valid_dataset.shape
(samples, width, height) = valid_dataset.shape
X_valid = np.reshape(valid_dataset, (samples, width*height))
y_valid = valid_labels

scorecard = []
for record in range(len(X_valid)):
   inputs = ((X_valid[record]+0.5)*0.98) +0.01
   outputs = n.query(inputs)
   label = numpy.argmax(outputs)
   correct_label = y_valid[record]
   if (label== correct_label):
        scorecard.append(1)
   else:
        scorecard.append(0)
        
scorecard_array = numpy.asarray(scorecard) 
print ("performance =", (scorecard_array.sum()*1.00/ scorecard_array.size*1.00))



