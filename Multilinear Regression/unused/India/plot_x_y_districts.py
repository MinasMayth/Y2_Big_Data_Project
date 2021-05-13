import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
import itertools

data = pd.read_csv('India_data_cleaned.csv')


print(data.head())
x = data['Population']
y = data['Confirmed']
z = data['Tested']
# d = data['Density'] #  No density data has been found
a = x
b = z

slope, intercept, r_value, p_value, std_err = stats.linregress(z,y)
print(slope, intercept, r_value, p_value, std_err)


print("correlation coefficient:",r_value)
print("p-value:",p_value)

fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = ax1.twinx()


ax1.scatter(z, y, color ='red', label='tests vs infections')

for x, y, label in zip(z, y, data['District']):
    ax1.text(x, y , s=label)

#ax2.scatter(a, z, color ='black', label='Tests')
#ax1.legend(loc=4)
#ax2.legend()
ax1.set_xlabel("No. of Tests")
ax1.set_ylabel("No. of Infections", color='red')
#ax2.set_ylabel("No. of Tests")
ax1.plot(z, intercept + slope*z, 'r', label='fitted line', color='black')
ax1.legend()
plt.show()
