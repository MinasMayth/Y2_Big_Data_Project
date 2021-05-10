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

dataUS = pd.read_csv(r'C:\Users\samya\Documents\Programming\Github_Repos\Y2_Big_Data_Project\Project Data\US States Data.csv')

def usa_features():
    return ['Population (discrete data)', 'Tests (discrete data)', 'Gini - gov 2019 (continuous data)',
            '% urban population (continuous data)', 'Actual cases (measured) (discrete data)']


def europe_features():
    return ['population (discrete data)', 'tests     (discrete data)', 'Gini      (discrete data)',
            '%urban pop.  (continuous data)', 'Actual cases']



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)


def feature_plot(array):
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


def MLtrain_x_test_y():
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X_train)
y_scaled = min_max_scaler.transform(y_train)
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(x_scaled)

print('Regression Intercept: \n', regr.intercept_)
print('Regression Coefficients: \n', regr.coef_)

 
model = sm.OLS(y_train, x_scaled).fit()
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


dataEU = pd.read_csv(r'C:\Users\samya\Documents\Programming\Github_Repos\Y2_Big_Data_Project\Project Data\europe.csv')

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