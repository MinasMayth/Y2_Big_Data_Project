import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm



data = pd.read_csv(r'C:\Users\samya\Documents\Github-Repos\Y2_Big_Data_Project\Multilinear Regression\US States Data.csv')

X = data.drop(['Predicted cases (tests & population)'])
Y = data['Actual cases (measured)']
