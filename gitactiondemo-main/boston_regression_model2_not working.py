import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from pandas import read_csv

boston = read_csv('./housing.csv')
print(boston.head(5))

dataset=pd.DataFrame(boston._data,columns=boston. feature_names)

print(dataset.head())

dataset['Price']=boston.target

print(dataset.head())

print(dataset.info())

print(dataset.describe())

## Check the missing Values
print(dataset.isnull().sum())

### EXploratory Data Analysis
## Correlation
print(dataset.corr())

import seaborn as sns
sns.pairplot(dataset)

#Analyzing The Correlated Features
print(dataset.corr())

plt.scatter(dataset['CRIM'],dataset['Price'])
plt.xlabel("Crime Rate")
plt.ylabel("Price")

plt.scatter(dataset['RM'],dataset['Price'])
plt.xlabel("RM")
plt.ylabel("Price")

