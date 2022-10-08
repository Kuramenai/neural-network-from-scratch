from cProfile import label
from operator import le
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import math


np.random.seed(0)

#ReLU Activation function
class ReLU:
  "Implementation of the Relu activation function and its derivative"

  def feedforward(self, inputs):
    "Calculate the activation function outputs"
    # We save the activation function inputs here 
    # cause we'll need them later when doing backpropagation
    self.inputs = inputs
    # Calculate the outputs by comparing 0 
    # and the inputs and getting the bigger one
    self.output = np.maximum(0, inputs)

    return self.output

  def derivative(self, dvalues):
    "Calculate the derivative of the ReLU function"
    # In order to get the gradients of the loss function 
    # with respect to the first layer weights and biases 
    # we need to get first the derivative of The ReLu in respect to its inputs W1.X + b1
    # dvalues: Gradients of the loss function with respect to ReLu outputs
    self.dinputs = dvalues.copy()
    # Gradients of the loss function with respect to  W1.X + b1
    self.dinputs[self.inputs <= 0] = 0

class Sigmoid():
    "Implementation of the Sigmoid activation function and its derivative"
    def feedforward(self,inputs):
        # Reason same as ReLu's 
        self.inputs = inputs
        self.output = 1/(1 + np.exp(-self.inputs))
    
    def derivative(self,dvalues):
        self.dinputs = self.output*(1 - self.output)
        self.dinputs = dvalues*self.dinputs

class Tanh():
    "Implementation of the tanh activation function and its derivative"
    def feedforward(self,inputs):
        # Reason same as ReLu's 
        self.inputs = inputs
        pos_exp = np.exp(self.inputs)
        neg_exp = np.exp(-self.inputs)
        self.output = (pos_exp - neg_exp)/(pos_exp + neg_exp)
    
    def derivative(self,dvalues):
        self.dinputs = 1 - np.square(self.output)
        self.dinputs = dvalues*self.dinputs




# Softmax activation
class Softmax:
  "Implementation of the Softmax activation function"
  #
  def feedforward(self, inputs):
    "Calculate the outputs values~probabilities for each class"
    # Reason same as ReLu's 
    self.inputs = inputs
    # In order to prevent explosive values 
    # we limit the input to exponential function by substracting each input
    #by the maximum input value of that sample
    exp_values = np.exp(inputs - np.max(inputs, axis=1,keepdims=True))
    #Normalization
    probabilities = exp_values / np.sum(exp_values, axis=1,keepdims=True)
    self.output = probabilities
    return probabilities


# Dense layer 
class DenseLayer:
  "A layer of of the neural network"

  def __init__(self, n_inputs, n_neurons):
    "Initialize the weights and biases of this layer"
    #To prevent the appearance of explosive gradients or vanishing ones 
    # We use the weight initialization method proposed by He

    # n_inputs : number of features
    # n_neurons : number of neurons in this layer
    self.weights = np.sqrt(2/(n_inputs + n_neurons))*np.random.randn(n_inputs,n_neurons)
    self.biases = np.zeros((1, n_neurons))

  def feedforward(self, inputs):
    "Calculate the result of the matrix multiplication between inputs and weights plus the biases"
    # Same reason as ReLU's
    self.inputs = inputs
    self.output = np.dot(inputs, self.weights) + self.biases

  def backpropagation(self, dvalues):
    """Calculate the gradients of the loss function
    with respect to this layer weightx,biases and inputs"""
    # dvalues : Gradients of the loss function 
    # obtained during the last chain rule calculation
    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
    self.dinputs = np.dot(dvalues, self.weights.T)


class NeuralNetwork:
    "A simple one hidden layer neural network"
    def __init__(self,no_inputs,no_neurons_per_layer,hidden_layer_activation_function = ReLU()):


        #Number of features
        self.no_inputs = no_inputs

        self.no_neurons_per_layer = no_neurons_per_layer

        #Create the hidden and output layers
        self.hidden_layer = DenseLayer(self.no_inputs,self.no_neurons_per_layer)
        self.output_layer = DenseLayer(self.no_neurons_per_layer,self.no_inputs)

        #Define the activation functions for the hidden layer and output layer respectively
        self.activation_function_hidden_layer = hidden_layer_activation_function
        self.activation_function_output_layer = Softmax()
    
    
    def feedforward(self,inputs,y_true):
        "Implementation of the feedforward process"

        self.samples_length = len(y_true)
 
        #Calculate the outputs of the hidden layer
        self.hidden_layer.feedforward(inputs)
        self.activation_function_hidden_layer.feedforward(self.hidden_layer.output)
    
        #Calculate the network predictions - output layer values
        self.output_layer.feedforward(self.activation_function_hidden_layer.output)
        self.activation_function_output_layer.feedforward(self.output_layer.output)

        self.calculate_loss(self.activation_function_output_layer.output,y_true)
    
    def calculate_predictions(self,y_true):
        "Calculate predictions and overall accuracy"
        self.predictions = np.argmax(self.activation_function_output_layer.output, axis=1)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.accuracy = np.mean(self.predictions==y_true)    
       
    
    def calculate_loss(self,y_pred,y_true):

        #Restrain the output values so that we won't have to deal with log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
           # For when labels are given in the form [1,0,1,0 1...]
            correct_confidences = y_pred_clipped[range(self.samples_length),y_true]
        elif len(y_true.shape) == 2:
            #For when labels are given in the form [[1, 0], [0, 1], ...]
            correct_confidences = np.sum(y_pred_clipped * y_true,axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        self.loss = np.mean(negative_log_likelihoods)
    
    def calculate_loss_function_gradients(self,y_true):
        """Calculate the gradients of the loss function 
        in respect to the inputs of the SoftMax function"""

        #If labels are given in the form [[1, 0], [0, 1], ...] flatten them
        if len(y_true.shape) == 2:
            self.y_true = np.argmax(y_true, axis=1)

        gradients = self.activation_function_output_layer.output.copy()

        # We calculate the gradients according to gradients =y_pred - y_true
        gradients[range(self.samples_length), y_true] -= 1
        #Normalization
        gradients = gradients / self.samples_length 
        return gradients

    def update_params(self, layer,learning_rate):

        weight_updates = -learning_rate * layer.dweights
        bias_updates = -learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates
    

    def backpropagagtion(self,learning_rate,y_true):
        "Implement the backpropagation process"
        
        loss_function_gradients = self.calculate_loss_function_gradients(y_true)

        self.output_layer.backpropagation(loss_function_gradients)
        self.activation_function_hidden_layer.derivative(self.output_layer.dinputs)
        self.hidden_layer.backpropagation(self.activation_function_hidden_layer.dinputs)

        self.update_params(self.hidden_layer,learning_rate)
        self.update_params(self.output_layer,learning_rate)
    
    def train(self,x_train,y_train,learning_rate,no_iterations):
        "Train the neural network"
        self.no_iterations = no_iterations
        self.loss_values = []
        self.accuracy_values = []
        
        for epoch in range(no_iterations):

            self.feedforward(x_train,y_train)

            self.calculate_predictions(y_train)

            self.loss_values.append(self.loss)
            self.accuracy_values.append(self.accuracy)

            if not epoch % 1000:
                print(f'epoch: {epoch}, ' +
                        f'acc: {self.accuracy:.3f}, ' +
                        f'loss: {self.loss:.3f}, ' +
                        f'lr: {learning_rate}')

            self.backpropagagtion(learning_rate,y_train)

    def test(self,x_test,y_test):
        self.feedforward(x_test,y_test)
        self.calculate_predictions(y_test)
        print(f'Performance of the neural network on the testing data \n' +
                        f'accuracy: {self.accuracy:.3f}, ' +
                        f'loss: {self.loss:.3f}\n')
    
    def draw_performance_charts(self):
        
        plt.xlabel("no_iterations")
        plt.ylabel("loss")
        plt.plot(range(0,self.no_iterations), self.loss_values)
        plt.show()

       
        plt.xlabel("no_iterations")
        plt.ylabel("accuracy")
        plt.plot(range(0,self.no_iterations), self.accuracy_values)
        plt.show()
    



#Load the data
half_moons_length_samples, half_moons_labels = datasets.make_moons(n_samples=10000,shuffle=True, noise=0.2, random_state=None)

# x_tain: Training smples
# y_train : Output labels for each training sample
# x_test: Testing samples to use after training
# y_test :  Output labels  for testing samples

x_train, x_test, y_train, y_test = train_test_split(half_moons_length_samples,half_moons_labels,test_size = 0.2)

no_input_features = 2
no_neurons_hidden_layer = 8

learning_rate = 0.01
training_iterations = 10001

#Build a one hidden layer neural network 
# with ReLU as the hidden layer activation function
MyNeuralNetwork = NeuralNetwork(no_input_features,no_neurons_hidden_layer)
MyNeuralNetwork.train(x_train,y_train,learning_rate,training_iterations)
MyNeuralNetwork.test(x_test,y_test)
loss_values_relu = MyNeuralNetwork.loss_values
accuracy_values_relu = MyNeuralNetwork.accuracy_values

#Build a second one, this time with Sigmoid as the hidden layer activation function
MyNeuralNetwork2 = NeuralNetwork(no_input_features,no_neurons_hidden_layer,Sigmoid())
MyNeuralNetwork2.train(x_train,y_train,learning_rate,training_iterations)
MyNeuralNetwork2.test(x_test,y_test)
loss_values_sigmoid = MyNeuralNetwork2.loss_values
accuracy_values_sigmoid = MyNeuralNetwork2.accuracy_values

#And a third one with Tanh as the hidden layer activation function
MyNeuralNetwork3 = NeuralNetwork(no_input_features,no_neurons_hidden_layer,Tanh())
MyNeuralNetwork3.train(x_train,y_train,learning_rate,training_iterations)
MyNeuralNetwork3.test(x_test,y_test)
loss_values_tanh = MyNeuralNetwork3.loss_values
accuracy_values_tanh = MyNeuralNetwork3.accuracy_values


#Plot the variation of loss for all of three networks 
# during the learning process
x_axis = list(range(0,10001))

plt.plot(x_axis,loss_values_relu,color='r',label='ReLU')
plt.plot(x_axis,loss_values_sigmoid,color='g',label='Sigmoid')
plt.plot(x_axis,loss_values_tanh,color='b',label='Tanh')

plt.xlabel('Number of iterations')
plt.ylabel('Loss')
plt.title("")

plt.legend()
plt.show()

#Plot the change in accuacy for all of three networks 
# during the learning process
plt.plot(x_axis,accuracy_values_relu,color='r',label='ReLU')
plt.plot(x_axis,accuracy_values_sigmoid,color='g',label='Sigmoid')
plt.plot(x_axis,accuracy_values_tanh,color='b',label='Tanh')

plt.xlabel('Number of iterations')
plt.ylabel('Accuracy')
plt.title("")

plt.legend()
plt.show()






