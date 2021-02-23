# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:58:08 2021

@author: samya
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


 # Contains population data
india_1 = pd.read_csv(r"Project Data\India_states_population_data.csv", error_bad_lines=False, infer_datetime_format=True)
 # Contains testing data
india_2 = pd.read_csv(r"Project Data\statewise_tested_numbers_data_INDIA.csv", error_bad_lines=False, infer_datetime_format=True)
 # Contains data for confirmed, recovered, deaths
india_3 = pd.read_csv(r"Project Data\state_wise_india.csv", error_bad_lines=False, infer_datetime_format=True)


illegal = ['2011', 'India', 'Total', 'State Unassigned']
state_pop = [(state['State/UT'], state['Population']) for i, state in india_1.iterrows() if state['State/UT'] not in illegal ]
state_infections = [(state['State'], state['Confirmed']) for i, state in india_3.iterrows() if state['State'] not in illegal]
state_tests = [(state['State'], state['Total Tested']) for i, state in india_2.iterrows() if state['Updated On'] == '14/02/2021']

state_values = []

for pstate in state_pop:
    if pstate[0] in [istate[0] for istate in state_infections] and pstate[0] in [tstate[0] for tstate in state_tests]:
        state_values.append(([pstate[0], float(pstate[1].replace(',', '')), float([i[1] for i in state_infections if i[0]==pstate[0]][0]), float([i[1] for i in state_tests if i[0]==pstate[0]][0])]))
        
        
#Initial graph
plt.scatter([x[1] for x in state_values], [x[2] for x in state_values], label="Confirmed Cases")

plt.xlabel('Population')
for x, y, label in zip([x[1] for x in state_values], [x[3] for x in state_values], [x[0] for x in state_values]):
    plt.text(x, y , s=label)
    
    
X_reshaped = np.array([x[1] for x in state_values]).reshape((-1,1))    

plt.scatter([x[1] for x in state_values], [x[3] for x in state_values], label="Total Tests")
plt.ylabel('Confirmed/Total Tested')

plt.legend()
plt.show()


ratios = [value[3]/value[2] for value in state_values]

plt.scatter([x[1] for x in state_values], ratios)

plt.xlabel('Population')

for x, y, label in zip([x[1] for x in state_values], ratios, [x[0] for x in state_values]):
    plt.text(x, y , s=label)
    
plt.ylabel('Total Tested/Confirmed RATIO')






model = LinearRegression().fit(X_reshaped, ratios)

r_sq = model.score(X_reshaped, ratios)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)


y_pred = model.predict(X_reshaped)
print('predicted response:', y_pred, sep='\n')



plt.plot(X_reshaped, y_pred)
plt.show()

"""
fields = ['State', 'Population', 'Confirmed Infections', 'Total Tested'] 


with open('india_combined_data.csv', 'w') as f: 
      
    # using csv.writer method from CSV package 
    write = csv.writer(f) 
      
    write.writerow(fields) 
    write.writerows(state_values) 

"""