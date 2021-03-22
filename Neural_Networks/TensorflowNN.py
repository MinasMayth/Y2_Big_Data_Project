import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# import talib
import random


# Reading the dataset
dataset = pd.read_csv(r'C:\Users\samya\Documents\Github-Repos\Y2_Big_Data_Project\Multilinear Regression\US States Data.csv')
dataset = dataset.dropna(axis=1, how='all')

data = dataset.drop(['States','Predicted cases (tests & population)'], axis=1)


print(data.columns)

#Dimensions of Data
n = data.shape[0]
p = data.shape[1]

data = data.values

print(data.shape)

X = data.drop('Actual cases (measured)')
y = data['Actual cases (measured)']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) # Test train splitting

scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)


n_features=X_train.shape[1]