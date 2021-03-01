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

owid_global_data = dataframes[4]

for date in range (30,31):
    current_clustering_data = owid_global_data.loc[(owid_global_data.date == '2021-01-'+str(date))]
    
    country_names = []
    country_ratios = []
    x_axis = []
    y_axis = []
    country_densities = []
    
    continents = ['North America','South America','Europe','Africa','Asia','World', 'Oceania']
    
    
    for i, country in current_clustering_data.iterrows():
        if country['location'] not in continents:
            local_ratio = country['total_tests']/country['population']
            country_names.append(country['location'])
            country_ratios.append(local_ratio)
            
            x_axis.append(country['population'])
            y_axis.append(country['total_tests'])
            
            country_densities.append(country['population_density'])
        
    
    
    countrys_tup = list(zip(country_names, country_ratios, country_densities))
    points = list(zip(x_axis,y_axis))
    
    #Removing all points that have invalid values
    country_tup_filtered = [value for value in countrys_tup if math.isnan(value[1]) == False]
    points_filtered = np.array([value for value in points if math.isnan(value[1]) == False])
    
    #Initial graph
    plt.scatter([x[0] for x in points_filtered], [x[1] for x in points_filtered])
    plt.xlabel('Population')
    plt.ylabel('Total Tests')
    for coordinate, label in zip(points_filtered, [country[0] for country in country_tup_filtered]):
        plt.annotate(xy=coordinate, text=label)
        
    plt.show()

    

# create kmeans object
kmeans = KMeans(n_clusters=4)
# fit kmeans object to data
kmeans.fit(points_filtered)
# print location of clusters learned by kmeans object
print(kmeans.cluster_centers_)
# save new clusters for chart
y_km = kmeans.fit_predict(points_filtered)   

#Without labels
plt.scatter(points_filtered[y_km ==0,0], points_filtered[y_km == 0,1], s=100, c='red')
plt.scatter(points_filtered[y_km ==1,0], points_filtered[y_km == 1,1], s=100, c='black')
plt.scatter(points_filtered[y_km ==2,0], points_filtered[y_km == 2,1], s=100, c='blue')
plt.scatter(points_filtered[y_km ==3,0], points_filtered[y_km == 3,1], s=100, c='cyan')
plt.xlabel('Population')
plt.ylabel('Total Tests')

plt.show()

#With labels
plt.scatter(points_filtered[y_km ==0,0], points_filtered[y_km == 0,1], s=100, c='red')
plt.scatter(points_filtered[y_km ==1,0], points_filtered[y_km == 1,1], s=100, c='black')
plt.scatter(points_filtered[y_km ==2,0], points_filtered[y_km == 2,1], s=100, c='blue')
plt.scatter(points_filtered[y_km ==3,0], points_filtered[y_km == 3,1], s=100, c='cyan')
plt.xlabel('Population')
plt.ylabel('Total Tests')


for coordinate, label in zip(points_filtered, [country[0] for country in country_tup_filtered]):
    plt.annotate(xy=coordinate, text=label)

plt.show()




plt.scatter([i[2] for i in country_tup_filtered], [i[1] for i in country_tup_filtered])
plt.xlabel('Population Density')
plt.ylabel('Tests/Population')

for x, y, label in zip([i[2] for i in country_tup_filtered], [i[1] for i in country_tup_filtered], [i[0] for i in country_tup_filtered]):
    plt.text(x,y,s=label)

plt.show