import pandas as pd
import numpy as np

us = pd.read_csv('USAclean.csv')
eu = pd.read_csv('EUclean.csv')
test = pd.read_csv('Test Data.csv')


us['Pop*1.1'] = us['Population (discrete data)']*1.1

print(us['Pop*1.1'])

us.to_csv('USAclean.csv')

eu['Pop*1.1'] = eu['population (discrete data)']*1.1

print(eu['Pop*1.1'])

eu.to_csv('EUclean.csv')

test['Pop*1.1'] = test['Population']*1.1

print(test['Pop*1.1'])

test.to_csv('Test Data.csv')