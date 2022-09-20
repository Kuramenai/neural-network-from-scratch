import numpy as np


#batch size of 3 
inputs = [[1, 2, 3, 2.5],
             [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]
        ]


#weights matrix of the first and second hidden layer
weights = [ [0.2, 0.8, -0.5, 1.0], 
            [0.5, -0.91, 0.26, -0.5], 
            [-0.26, -0.27, 0.17, 0.87]]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

#bias vector containing the biases for each neuron in the first and second layer
biases = [2, 3, 0.5]

biases2 = [-1, 2, -0.5]


#calculating value of the neurons for the first and second layer for every sample
layer1_outputs = np.dot(inputs,np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2


print(layer1_outputs)
print(layer2_outputs)