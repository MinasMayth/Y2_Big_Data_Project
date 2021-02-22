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

india_1 = dataframes[4]
india_2 = dataframes[5]