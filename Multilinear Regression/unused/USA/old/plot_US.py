import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import itertools
import numpy as np

data = pd.read_csv(r'C:\Users\samya\Documents\Programming\Github_Repos\Y2_Big_Data_Project\USA\state cases and tests.csv')


print(data.head())
x = np.array(data['Population']).reshape(-1,1)
y = np.array(data['Cumulative cases'])
z = np.array(data['Cumulative tests'])
# d = data['Density']
a = x
b = y


reg = LinearRegression().fit(x, y)
print("correlation coefficient:",reg.coef_ )

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()


ax1.scatter(a, y, color ='red', label='Infections')
ax2.scatter(a, z, color ='black', label='Tests')
for ee, bee, label in zip(a, z, data['State']):
    ax2.text(ee, bee , s=label)
ax1.legend(loc=4)
ax2.legend()
ax1.set_xlabel("Population")
ax1.set_ylabel("Cumulative Cases", color='red')
ax2.set_ylabel("Cumulative Tests", color='black')
plt.show()
