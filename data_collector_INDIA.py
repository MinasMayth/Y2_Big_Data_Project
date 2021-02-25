# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:58:08 2021

@author: samya
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

### DATA COLLECTION SECTION ###

 # Contains population and population density data
india_pop_csv = pd.read_csv(r"Project Data\India_states_population_data.csv", error_bad_lines=False, infer_datetime_format=True)

 # Contains testing data
india_tests_csv = pd.read_csv(r"Project Data\statewise_tested_numbers_data_INDIA.csv", error_bad_lines=False, infer_datetime_format=True)

 # Contains data for confirmed, recovered, deaths
india_cases_csv = pd.read_csv(r"Project Data\state_wise_india.csv", error_bad_lines=False, infer_datetime_format=True)

 # These are values contained in the data that do not represent a specific state - therefore they are cleaned from it
illegal = ['2011', 'India', 'Total', 'State Unassigned']

 # A list of each state with its corresponding population density
state_pop = [(state['State/UT'], state['Population']) for i, state in india_pop_csv.iterrows() if state['State/UT'] not in illegal ]

 # A list of each state with total test number for 14/02/21 - Chose this date because it is comparatively
 # close to the present and has relatively complete data
state_tests = [(state['State'], state['Total Tested']) for i, state in india_tests_csv.iterrows() if state['Updated On'] == '14/02/2021']

 # List of states with coressponding confirmed cases
state_infections = [(state['State'], state['Confirmed']) for i, state in india_cases_csv.iterrows() if state['State'] not in illegal]

 # Initializes state_values, this is where we will store all the data together
state_values = []

 # Iterates through the population list, and if a corresponding state value is found in both
 # other lists, they are appended to state_values.
 # We also clean commas out of the data here and convert values to float.
for pstate in state_pop:
    if pstate[0] in [istate[0] for istate in state_infections] and pstate[0] in [tstate[0] for tstate in state_tests]:
        state_values.append(([pstate[0], float(pstate[1].replace(',', '')), float([i[1] for i in state_infections if i[0]==pstate[0]][0]), float([i[1] for i in state_tests if i[0]==pstate[0]][0])]))



### PLOTTING SECTION ###        

 # Scatter of pop vs confirmed cases followed by pop vs total tests
plt.scatter([x[1] for x in state_values], [x[2] for x in state_values], label="Confirmed Cases")
        
plt.scatter([x[1] for x in state_values], [x[3] for x in state_values], label="Total Tests")

 # Labels individual points on the graph with their state names
for x, y, label in zip([x[1] for x in state_values], [x[3] for x in state_values], [x[0] for x in state_values]):
    plt.text(x, y , s=label)

plt.xlabel('Population')
plt.ylabel('Confirmed/Total Tested')
plt.legend()
plt.show()


 # Calculating total tested/confirmed cases for each state value
ratios = [value[3]/value[2] for value in state_values]

 # Scatter plot of pop vs ratios - we want to be able to use pop density here
plt.scatter([x[1] for x in state_values], ratios)

 # Labelling points
for x, y, label in zip([x[1] for x in state_values], ratios, [x[0] for x in state_values]):
    plt.text(x, y , s=label)
    
 
plt.xlabel('Population')
plt.ylabel('Total Tested/Confirmed RATIO')


### MACHINE LEARNING SECTION ###

 # Reshaping the population to use for linear regression
X_reshaped = np.array([x[1] for x in state_values]).reshape((-1,1))


 # creates a linear regression model and fits it
model = LinearRegression().fit(X_reshaped, ratios)
r_sq = model.score(X_reshaped, ratios)

 # Value printing
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

 # Predictions
y_pred = model.predict(X_reshaped)
print('predicted response:', y_pred, sep='\n')

 # Plotting the predictions
plt.plot([x[1] for x in state_values], y_pred)
plt.show()





"""
 # Used for writing data to csv
fields = ['State', 'Population', 'Confirmed Infections', 'Total Tested'] 


with open('india_combined_data.csv', 'w') as f: 
      
    # using csv.writer method from CSV package 
    write = csv.writer(f) 
      
    write.writerow(fields) 
    write.writerows(state_values) 

"""