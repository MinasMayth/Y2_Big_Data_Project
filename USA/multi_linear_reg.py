# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 11:08:37 2021

@author: samya
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv(r'C:\Users\samya\Documents\Programming\Github_Repos\Y2_Big_Data_Project\USA\state cases and tests.csv')

X = np.array(data[['Population', 'Cumulative tests']]) # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = np.array(data['Cumulative cases'])
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

plt.subplot(2,2,1)
plt.scatter(X[:,0],Y)
plt.subplot(2,2,2)
plt.scatter(X[:,1],Y)


print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x1 = X[:,0]
x2 = X[:,1]
ax.scatter(x1, x2, Y, c='r', marker='o')
# Set axis labels
ax.set_xlabel('Population')
ax.set_ylabel('Cumulative Tests')
ax.set_zlabel('Cumulative Cases')