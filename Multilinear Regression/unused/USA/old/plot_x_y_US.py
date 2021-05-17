import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import itertools
import numpy as np

data = pd.read_csv(r'C:\Users\samya\Documents\Programming\Github_Repos\Y2_Big_Data_Project\USA\state cases and tests.csv')

data = data.dropna(how="all")
data = data.dropna(axis=1)
print(data.head())

x = np.array(data['Population'])
y = np.array(data['Cumulative cases'])
z = np.array(data['Cumulative tests']).reshape(-1,1)
# d = data['Density'] #  No density data has been found
a = x
b = z

#slope, intercept, r_value, p_value, std_err = stats.linregress(z,y)
#print(slope, intercept, r_value, p_value, std_err)

reg = LinearRegression().fit(z, y)

print("correlation coefficient:",reg.coef_ )
#print("p-value:",p_value)

fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = ax1.twinx()


ax1.scatter(z, y, color ='red', label='tests vs infections')

for x, y, label in zip(z, y, data['State']):
    ax1.text(x, y , s=label)

#ax2.scatter(a, z, color ='black', label='Tests')
#ax1.legend(loc=4)
#ax2.legend()
ax1.set_xlabel("No. of Tests")
ax1.set_ylabel("No. of Infections", color='red')
#ax2.set_ylabel("No. of Tests")
ax1.plot(z, reg.predict(z), label='fitted line', color='black')
ax1.legend()
plt.show()
