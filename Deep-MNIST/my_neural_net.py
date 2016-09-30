
# coding: utf-8

# In[16]:

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


# In[44]:

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.1

#Create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


# In[18]:

# load the mnist training data CSV file into a list
training_data_file = open("C:\Users\s6324900\Desktop\Deep learning\my_neural_net\mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()


# In[20]:

# train the neural network
#go through all records in the training data set 
for record in training_data_list:
    #split the record by the',' commas 
    all_values = record.split(',')
    #scale and shift the inputs 
    inputs = (numpy.asfarray(all_values[1:])/ 255.0 * 0.99) + 0.01
    #create the target output values (all 0.01, except the desired label which is 0.99)
    targets = numpy.zeros(output_nodes) + 0.01

    #all_values[0] is the target label for this record 
    targets[int(all_values[0])] = 0.99

    # train neural network
    n.train(inputs,targets)
    pass


# In[21]:

#load the mnist test data CSV file into a list
test_data_file = open("C:\Users\s6324900\Desktop\Deep learning\my_neural_net\mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines() 
test_data_file.close()


# In[25]:

#test the neural network
#scorecard for how well the network performs, initially empty 
scorecard = []

#go through all the records in the test data set 
for record in test_data_list:
    #split the record by the',' commas 
    all_values = record.split(',')
    #correct answer is first value 
    correct_label = int(all_values[0])
    print(correct_label, "correct label")
    #scale and shift the inputs 
    inputs = (numpy.asfarray(all_values[1:])/ 255.0 * 0.99) + 0.01
    #query the network 
    outputs = n.query(inputs)
    #the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    print(label, "network's answer")
    #append correct or incorrect to list
    if (label == correct_label):
        #network's answer matches correct answer, add 1 to scorecard 
        scorecard.append(1)
    else:
        #network's answer doesn't match correct answer, add 0 to scorecard 
        scorecard.append(0)
        pass
    pass


# In[28]:

print (scorecard)


# In[38]:

#calculate the performance score, the fraction of correct answers 
scorecard_array = numpy.asarray(scorecard) 
print scorecard_array
print ("performance =", (scorecard_array.sum()*1.00/ scorecard_array.size*1.00))


# In[39]:

# Now run on all data
# load the mnist training data CSV file into a list
training_data_file = open("C:\Users\s6324900\Desktop\Deep learning\my_neural_net\mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()


# In[45]:

# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 5

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass


# In[46]:

# load the mnist test data CSV file into a list
test_data_file = open("C:\Users\s6324900\Desktop\Deep learning\my_neural_net\mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()


# In[47]:

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    
    pass


# In[48]:

#calculate the performance score, the fraction of correct answers 
scorecard_array = numpy.asarray(scorecard) 
print ("performance =", (scorecard_array.sum()*1.00/ scorecard_array.size*1.00))


# In[ ]:



