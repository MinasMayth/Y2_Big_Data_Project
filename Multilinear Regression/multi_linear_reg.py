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



def graph_plotter(Xvar, Yvar):
    X = Xvar.values.reshape(-1,2)
    Y = Yvar
    
    # Create range for each dimension
    x = X[:, 0]
    y = X[:, 1]
    z = Y
    
    xx_pred = np.linspace(0, max(X[:,1]), 30)  # range of price values
    yy_pred = np.linspace(0, max(Y), 30)  # range of advertising values
    xx_pred, yy_pred = np.meshgrid(xx_pred, yy_pred)
    model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T
    
    # Predict using model built on previous step
    # ols = linear_model.LinearRegression()
    # model = ols.fit(X, Y)
    predicted = model.predict(model_viz)
    
    # Evaluate model by using it's R^2 score 
    r2 = model.rsquared
    
    # Plot model visualization
    plt.style.use('fivethirtyeight')
    
    fig = plt.figure(figsize=(12, 4))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    
    axes = [ax1, ax2, ax3]
    
    for ax in axes:
        ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
        ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
        ax.set_xlabel('Population', fontsize=12)
        ax.set_ylabel('Cumulative tests', fontsize=12)
        ax.set_zlabel('Cumulative cases', fontsize=12)
        ax.locator_params(nbins=4, axis='x')
        ax.locator_params(nbins=5, axis='x')
    
    ax1.view_init(elev=25, azim=-60)
    ax2.view_init(elev=15, azim=15)
    ax3.view_init(elev=25, azim=60)
    
    fig.suptitle('Multi-Linear Regression Model Visualization ($R^2 = %.2f$)' % r2, fontsize=15, color='k')
    
    fig.tight_layout()



data = pd.read_csv(r'C:\Users\samya\Documents\Github-Repos\Y2_Big_Data_Project\Multilinear Regression\US States Data.csv')

X = data[['Population', 'Tests']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = data['Actual cases (measured)']
 



# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

plt.subplot(2,2,1)
plt.scatter(np.array(X)[:,0],Y)
plt.subplot(2,2,2)
plt.scatter(np.array(X)[:,1],Y)
plt.show()


print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

 
model = sm.OLS(Y, X).fit()

print_model = model.summary()
print(print_model)

print(graph_plotter(X, Y))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

X = np.array(X)
Y = np.array(Y)

ax1.set_xlabel('Population')
ax1.scatter(X[:,0], Y, color ='red', label='Infections')
ax1.set_ylabel('Infections')
ax2.scatter(X[:,1], Y, color ='black', label='Tests')
ax2.set_ylabel('Tests')
plt.show()

predictions1 = model.predict(X.astype(int))



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x1 = X[:,0]
x2 = X[:,1]
ax.scatter(x1, x2, Y, c='r', marker='o')
# Set axis labels
ax.set_xlabel('Population', fontsize=10)
ax.set_ylabel('Cumulative Tests', fontsize=10)
ax.set_zlabel('Cumulative Cases', fontsize=10)
fig.suptitle('US Pop vs Tests vs Cases')
plt.show()



dataEU = pd.read_csv(r'C:\Users\samya\Documents\Github-Repos\Y2_Big_Data_Project\Multilinear Regression\europe.csv')

dataEU = dataEU.replace(',','', regex=True)

#X2 = dataEU[['country', 'tests', 'population', 'Actual cases']]

X2 = np.array(dataEU[['population', 'tests']])
Y2 = dataEU['Actual cases (measured)']

predictions = model.predict(X2.astype(int))

print(predictions)

fig = plt.figure() 
ax = fig.gca(projection ='3d') 
  
ax.scatter(X2[:, 0].astype(int), X2[:,1].astype(int), Y2.astype(int), label ='y') 
ax.scatter(X2[:, 0].astype(int), X2[:,1].astype(int), predictions.astype(int), label='predicted')
ax.legend()
ax.set_xlabel('Population', fontsize=10)
ax.set_ylabel('Tests', fontsize=10)
ax.set_zlabel('Actual cases', fontsize=10)
fig.suptitle('US Model on European Data')
plt.show() 

