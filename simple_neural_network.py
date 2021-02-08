# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:15:34 2021

This is a simple neural network following the instructions from:
    https://medium.com/better-programming/how-to-create-a-simple-neural-network-in-python-dbf17f729fe6
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

# Here we define the neural network

class NeuralNetwork():
    
    
    
    def __init__(self,):
        """
        The __init__ function will initialize the variables we need for working
        with the neural network when the class is first created.
        This neural network has three input nodes, three nodes in the hidden 
        layer, and one output node.
        """
        self.inputSize = 3 # inputSize is the number of input nodes, which should be equal to the number of features in our input data
        self.outputSize = 1 # equal to the number of output nodes
        self.hiddenSize = 3 # the number of nodes in the hidden layer
        
        #W1 and W2 are weights between the different nodes in our network that will be adjusted during training.
        self.W1 = np.random.rand(self.inputSize, self.hiddenSize)
        self.W2 = np.random.rand(self.hiddenSize, self.outputSize)
        
        self.error_list = [] # will contain the mean absolute error (MAE) for each of the epochs
        self.limit = 0.5 # will describe the boundary for when a vector should be classified as a vector with element 10 as the first element and not
        
        # variables that will be used to store the number of true positives, false positives, true negatives, and false negatives.
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        
        
    
    def forward(self, X):
        """
        The purpose of the forward pass function is to iterate forward through the different layers
        of the neural network to predict output for that particular epoch. Then, looking at the 
        difference between the predicted output and the actual output, the weights will be updated 
        during backward propagation.
        """
        self.z = np.matmul(X, self.W1) # the values at the nodes in the previous layer will be matrix multiplied with the applicable weights
        self.z2 = self.sigmoid(self.z) # a non-linear activation function will be applied to widen the possibilities for the final output function. 
        self.z3 = np.matmul(self.z2, self.W2)
        o = self.sigmoid(self.z3) # In this example, we have chosen the Sigmoid as the activation function, but there are also many other alternatives.
        return o
    
    
    # Sigmoid Activation function
    def sigmoid(self,s):
        return 1/(1 + np.exp(-s))
    
    def sigmoidPrime(self,s):
        return s*(1-s)
    
    def backward(self,X,y,o):
        """
        Backpropagation is the process that updates the weights for the different nodes in the neural 
        network and hence decides their importance. The output error from the output layer is calculated 
        as the difference between the predicted output from forwarding propagation and the actual output.
        Then, this error is multiplied with the Sigmoid prime in order to run gradient descent, before 
        the entire process is repeated until the input layer is reached. Finally, the weights between 
        the different layers are updated.
        """
        self.o_error = y - o #Output error is calculated
        self.o_delta = self.o_error * self.sigmoidPrime(o) #Multiplied by sigmoid prime
        self.z2_error = np.matmul(self.o_delta,
                                  np.matrix.transpose(self.W2)) #Matrix multiplication again
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2) #Multiplied by sigmoid prime
        self.W1 += np.matmul(np.matrix.transpose(X), self.z2_delta) #W1 Updated
        self.W2 += np.matmul(np.matrix.transpose(self.z2), 
                             self.o_delta) #W2 Updated
        
    def train(self,X,y,epochs):
        """
        During training, the algorithm will run forward and backward pass and thereby updating the 
        weights as many times as there are epochs. This is necessary in order to end up with the 
        most precise weights. In addition to running forward and backward pass, we save the mean 
        absolute error (MAE) to an error list so that we can later observe how the mean absolute 
        error develops during the course of the training.
        """
        for epoch in range(epochs):
            o = self.forward(X) #Forward pass
            self.backward(X,y,o) #Backward Pass
            self.error_list.append(np.abs(self.o_error).mean())
            
    def predict(self, x_predicted):
        """
        After the weights are fine-tuned during training, the algorithm is ready to predict the output 
        for new data points. This is done through a single iteration of forwarding pass. The predicted 
        output will be a number that hopefully will be quite close to the actual output.
        """
        return self.forward(x_predicted).item()
    
    def view_error_development(self):
        """
        There are many ways to evaluate the quality of a machine learning algorithm. One of the measures
        that are often used is the mean absolute error, and this should decrease with the number of epochs.
        """
        plt.plot(range(len(self.error_list)), self.error_list)
        plt.title("Mean Sum Squared Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
    def test_evaluation(self, input_test, output_test):
        """
        The number of true positives, false positives, true negatives, and false negatives describes 
        the quality of a machine learning classification algorithm. After training the neural network, 
        the weights should be updated so that the algorithm is able to accurately predict new data 
        points. In binary classification tasks, these new data points can only be 1 or 0. Depending 
        on whether the predicted value is above or below the defined limit, the algorithm will classify 
        the new entry as 1 or 0.

        """
        for i, test_element in enumerate(input_test):
            if self.predict(test_element) > self.limit and \
                output_test[i] == 1:
                    self.true_positives += 1
            if self.predict(test_element) < self.limit and \
                output_test[i] == 1:
                    self.false_negatives += 1
            if self.predict(test_element) > self.limit and \
                output_test[i] == 0:
                    self.false_positives += 1
            if self.predict(test_element) < self.limit and \
                output_test[i] == 0:
                    self.true_negatives += 1
        print('True positives: ', self.true_positives,
              '\nTrue negatives: ', self.true_negatives,
              '\nFalse positives: ', self.false_positives,
              '\nFalse negatives: ', self.false_negatives,
              '\nAccuracy: ',
              (self.true_positives + self.true_negatives) /
              (self.true_positives + self.true_negatives +
               self.false_positives + self.false_negatives))
        
    def rmse_calculator(self, input_test, output_test):
        """
        Calculator for root mean square error
        """
        actual = output_test
        pred = [self.predict(i) for i in input_test]
        
        #calculate RMSE
        rmse = sqrt(mean_squared_error(actual, pred)) 
        return rmse

if __name__ == "__main__":
    """
    To make sure that the model is evaluated based on how good 
    it is to predict new data points, and not how well it is 
    modeled to the current ones, it is common to split the 
    datasets into one training set and one test set (and 
    sometimes a validation set).
    """
    
    input_train = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0], 
                            [10, 0, 0], [10, 1, 1], [10, 0, 1]])
    
    output_train = np.array([[0], [0], [0], [1], [1], [1]])
    
    input_pred = np.array([1, 1, 0])
    
    input_test = np.array([[1, 1, 1], [10, 0, 1], [0, 1, 10], 
                           [10, 1, 10], [0, 0, 0], [0, 1, 1]])
    
    output_test = np.array([[0], [1], [0], [1], [0], [0]])
    
    """
    This MinMaxScaler scales and translates each feature individually 
    such that it is in the given range on the training set, e.g. 
    between zero and one. 
    """
    
    scaler = MinMaxScaler()
    input_train_scaled = scaler.fit_transform(input_train)
    output_train_scaled = scaler.fit_transform(output_train)
    input_test_scaled = scaler.fit_transform(input_test)
    output_test_scaled = scaler.fit_transform(output_test)
    
    
    
    NN = NeuralNetwork()
    NN.train(input_train_scaled, output_train_scaled, 2000)
    print(NN.predict(input_pred))
    NN.view_error_development()
    NN.test_evaluation(input_test_scaled, output_test_scaled)
    print(NN.rmse_calculator(input_test_scaled, output_test_scaled))
    
