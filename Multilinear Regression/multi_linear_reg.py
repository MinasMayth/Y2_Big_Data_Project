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
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


"""
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

"""

dataUS = pd.read_csv(r'C:\Users\samya\Documents\Github-Repos\Y2_Big_Data_Project\Multilinear Regression\US States Data.csv')

X = dataUS[['Population', 'Tests', 'Gini - gov 2019', '% urban population']]  # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = data['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = dataUS['Actual cases (measured)']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)


fig = plt.figure()
ax1 = plt.subplot(2,2,1)
ax1.scatter(np.array(X)[:,0],Y)
ax1.title.set_text("Population vs Actual cases")
ax2 = plt.subplot(2,2,2)
ax2.scatter(np.array(X)[:,1],Y)
ax2.title.set_text("Tests vs Actual cases")
ax3 = plt.subplot(2,2,3)
ax3.scatter(np.array(X)[:,2],Y)
ax3.title.set_text("Gini vs Actual cases")
ax4 = plt.subplot(2,2,4)
ax4.scatter(np.array(X)[:,3],Y)
ax4.title.set_text("% Urban population vs Actual cases")
plt.show()

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

print('Regression Intercept: \n', regr.intercept_)
print('Regression Coefficients: \n', regr.coef_)

 
model = sm.OLS(y_train, X_train).fit()
print_model = model.summary()
print(print_model)


predictions1 = regr.predict(X_test.astype(int))  # SKlearn model predictions
predictions2 = model.predict(X_test.astype(int))  # OLS model predictions

fig = plt.figure()
ax = sns.regplot(x=predictions1, y=y_test, ci=None, color="b")
ax.title.set_text("SKlearn prediction results vs Actual cases (USA)")
ax.set_xlabel("SKLearn prediction results")
ax.set_ylabel("Actual Cases (USA)")
plt.show()

fig = plt.figure()
ax = sns.regplot(x=predictions2, y=y_test, ci=None, color="b")
ax.title.set_text("OLS prediction results vs Actual cases (USA)")
ax.set_xlabel("OLS prediction results")
ax.set_ylabel("Actual Cases (USA)")
plt.show()


dataEU = pd.read_csv(r'C:\Users\samya\Documents\Github-Repos\Y2_Big_Data_Project\Multilinear Regression\europe.csv')

dataEU = dataEU.replace(',','', regex=True)

X2 = np.array(dataEU[['population', 'tests', 'Gini', '%urban pop.']])
Y2 = dataEU['Actual cases (measured)']

EUpredictions1 = regr.predict(X2.astype(int))
EUpredictions2 = model.predict(X2.astype(int))


fig = plt.figure()
ax = sns.regplot(x=EUpredictions1, y=Y2.astype(float), ci=None, color="b")
ax.title.set_text("SKLearn prediction results vs Actual cases (EU)" )
ax.set_xlabel("OLS prediction results")
ax.set_ylabel("Actual Cases (EU)")
plt.show()

fig = plt.figure()
ax = sns.regplot(x=EUpredictions2, y=Y2.astype(float), ci=None, color="b")
ax.title.set_text("OLS prediction results vs Actual cases (EU)")
ax.set_xlabel("OLS prediction results")
ax.set_ylabel("Actual Cases (EU)")
plt.show()