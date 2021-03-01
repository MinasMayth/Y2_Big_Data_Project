import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_csv('India_data_cleaned.csv')


print(data.head())
x = data['Population']
y = data['Confirmed']
z = data['Tested']
# d = data['Density']
a = x
b = z


slope, intercept, r_value, p_value, std_err = stats.linregress(a,y)
print(slope, intercept, r_value, p_value, std_err)


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()


ax1.scatter(a, y, color ='red', label='Infections')
ax2.scatter(a, z, color ='black', label='Tests')
for ee, bee, label in zip(a, z, data['District']):
    ax2.text(ee, bee , s=label)
ax1.legend(loc=4)
ax2.legend()
ax1.set_xlabel("Population")
ax1.set_ylabel("No. of Infections", color='red')
ax2.set_ylabel("No. of Tests", color='black')
plt.show()
