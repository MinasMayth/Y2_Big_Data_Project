# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:15:34 2021

This is a simple neural network following the instructions from:
    https://medium.com/better-programming/how-to-create-a-simple-neural-network-in-python-dbf17f729fe6
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.model_selection import train_test_split

#working data file
data = pd.read_csv("owid-covid-data.csv")


# Here we define the neural network

class NeuralNetwork():
    
    
    
    def __init__(self,):
        """
        The __init__ function will initialize the variables we need for working
        with the neural network when the class is first created.
        This neural network has three input nodes, three nodes in the hidden 
        layer, and one output node.
        """
        self.inputSize = 2 # inputSize is the number of input nodes, which should be equal to the number of features in our input data
        self.outputSize = 1 # equal to the number of output nodes
        self.hiddenSize = 9 # the number of nodes in the hidden layer
        
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
        self.Z1 = np.matmul(X, self.W1) # the values at the nodes in the previous layer will be matrix multiplied with the applicable weights
        self.A1 = self.sigmoid(self.Z1) # a non-linear activation function will be applied to widen the possibilities for the final output function. 
        self.Z2 = np.matmul(self.A1, self.W2)
        o = self.sigmoid(self.Z2) # In this example, we have chosen the Sigmoid as the activation function, but there are also many other alternatives.
        return o
    
    
    # Sigmoid Activation function - we can think of adding other ones here!
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
        
        self.A1_error = np.matmul(self.o_delta, np.matrix.transpose(self.W2)) #multiplication by the tranpose of W2
        self.A1_delta = self.A1_error * self.sigmoidPrime(self.A1) #Multiplied by sigmoid prime

        self.W1 += np.matmul(np.matrix.transpose(X), self.A1_delta) #W1 Updated
        self.W2 += np.matmul(np.matrix.transpose(self.A1), self.o_delta) #W2 Updated
        
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
            if self.predict(test_element) > self.limit and output_test[i] == 1:
                    self.true_positives += 1
            if self.predict(test_element) < self.limit and output_test[i] == 1:
                    self.false_negatives += 1
            if self.predict(test_element) > self.limit and output_test[i] == 0:
                    self.false_positives += 1
            if self.predict(test_element) < self.limit and output_test[i] == 0:
                    self.true_negatives += 1
        print('True positives: ', self.true_positives,
              '\nTrue negatives: ', self.true_negatives,
              '\nFalse positives: ', self.false_positives,
              '\nFalse negatives: ', self.false_negatives,
              '\nAccuracy: ',
              (self.true_positives + self.true_negatives) /
              (self.true_positives + self.true_negatives +
               self.false_positives + self.false_negatives))




if __name__ == "__main__":
    
    country_code = input("Enter Country code: ").upper()
    
    data2 = data[(data['iso_code'] == country_code)]
    
    for x in data2.index:
        if pd.isnull(data2.loc[x, 'new_cases']) or pd.isnull(data2.loc[x, 'new_tests']) or pd.isnull(data2.loc[x, 'hosp_patients']):
            data2.drop(x, inplace = True)
            
    X_orig = data2['date']
    y = np.array(data2['new_cases']).reshape(-1,1)
    y2 = np.array([[i for i in data2['new_tests']], [i for i in data2['hosp_patients']]])
    
    X_new = pd.to_datetime(X_orig)
    X_new = X_new.map(dt.datetime.toordinal)
    X_new = np.sort(np.array(X_new).astype(np.float64)).reshape(-1,1)
    
    
    #Plotting Overall graph for overview- no regression is done here
    fig, ax = plt.subplots(figsize=(10,7.5))
    ax.plot_date(X_orig, y, xdate=True,color='red', label = "No. of cases")
    ax.plot_date(X_orig,y2[1],color='green', label = "No. of hospital patients")
    ax.plot_date(X_orig,y2[0],color='blue', label = "No. of tests")
    fig.autofmt_xdate()
    ax.set_xlim()
    ax.set_ylim(bottom=0)
    plt.xticks(np.arange(0, len(X_orig), step=20))
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of People")
    ax.legend()
    plt.title("Number of hospital patients vs Number of new tests vs Number of cases on a Time Scale")
    plt.show()
    
    plt.scatter(y2[0],y2[1])
    plt.show()
    plt.title("Number of hospital patients vs Number of new tests")
    
    y2 = y2.transpose()
    
    input_train, input_test, output_train, output_test = train_test_split(y2, y, test_size=0.33)

    
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
    NN.train(input_train_scaled, output_train_scaled, 20)
    NN.view_error_development()
    NN.predict([11896overflow encountered in exp
  return 1/(1 + np.exp(-s)),12597])
    #NN.test_evaluation(input_test_scaled, output_test_scaled)
    
