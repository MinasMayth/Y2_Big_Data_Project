# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:58:08 2021

@author: samya
"""

import glob
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import KMeans
from sklearn.cluster import KMeans

# the path to your csv file directory
mycsvdir = (r'C:\Users\samya\Documents\Programming\Github_Repos\Y2_Big_Data_Project\Project Data')

# get all the csv files in that directory (assuming they have the extension .csv)
files = glob.glob(os.path.join(mycsvdir, '*.csv'))

# loop through the files and read them in with pandas
dataframes = []  # a list to hold all the individual pandas DataFrames
for file in files:
    if file.endswith('.csv'):
        df = pd.read_csv(file, error_bad_lines=False, infer_datetime_format=True)
        dataframes.append(df)

india_1 = dataframes[1]
india_2 = dataframes[5]
india_3 = dataframes[6]


illegal = ['2011', 'India', 'Total', 'State Unassigned']
state_pop = [(state['State/UT'], state['Population']) for i, state in india_1.iterrows() if state['State/UT'] not in illegal ]
state_infections = [(state['State'], state['Confirmed']) for i, state in india_3.iterrows() if state['State'] not in illegal]
state_tests = [(state['State'], state['Total Tested']) for i, state in india_2.iterrows() if state['Updated On'] == '14/02/2021']

state_values = []

for pstate in state_pop:
    if pstate[0] in [istate[0] for istate in state_infections] and pstate[0] in [tstate[0] for tstate in state_tests]:
        state_values.append(([pstate[0], pstate[1], [i[1] for i in state_infections if i[0]==pstate[0]][0], [i[1] for i in state_tests if i[0]==pstate[0]][0]]))
        
        
#Initial graph
plt.scatter([x[1] for x in state_values], [x[2] for x in state_values])
plt.xlabel('Population')
plt.ylabel('Confirmed Infections')
for x, y, label in zip([x[1] for x in state_values], [x[2] for x in state_values], [x[0] for x in state_values]):
    plt.text(x, y , s=label)
        
plt.show()


#Initial graph
plt.scatter([x[1] for x in state_values], [x[3] for x in state_values])
plt.xlabel('Population')
plt.ylabel('Total Tested')
for x, y, label in zip([x[1] for x in state_values], [x[3] for x in state_values], [x[0] for x in state_values]):
    plt.text(x, y , s=label)
        
plt.show()


