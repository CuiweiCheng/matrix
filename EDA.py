
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import pandas_datareader.data as wb
import pandas as pd
import numpy as np
import seaborn as sns


# 1.import data
start = '2016-01-01'
end = '2018-11-30'

aapl = wb.DataReader('AAPL', 'yahoo', start, end)
tsla = wb.DataReader('TSLA', 'yahoo', start, end)
gs = wb.DataReader('GS', 'yahoo', start, end)
ms = wb.DataReader('MS', 'yahoo', start, end)

# alternativly do as a panel object
df = wb.DataReader(['AAPL','TSLA','GS','MS'], 'yahoo', start, end)


# 2.get a basic decription
# 2.1 describe the data
df.info()
df.describe()

aapl.head()
aapl.tail()
aapl.info()
aapl.describe()

tsla.head()
tsla.tail()
tsla.info()
tsla.describe()

gs.head()
gs.tail()
gs.info()
gs.describe()

ms.head()
ms.tail()
ms.info()
ms.describe()

# 2.2 deal with missing value
# interpolate missing values
# the interpolate() function will perform a linear interpolation at the missing data points to “guess” the value that is most likely to be filled in.
aapl.interpolate()
tsla.interpolate()
gs.interpolate()
ms.interpolate()

# Visulaize time series data
# Plot the closing prices for `aapl`
aapl['Close'].plot(grid=True)
plt.show()

tsla['Close'].plot(grid=True)
plt.show()

gs['Close'].plot(grid=True)
plt.show()

ms['Close'].plot(grid=True)
plt.show()

# create a line plot showing Close price for these four stocks
tickers = ['AAPL', 'TSLA', 'MS', 'GS']
for tick in tickers:
    df['Adj Close'][tick].plot(figsize=(12,4),label=tick)
plt.legend()

# plot the rolling 20 days moving average for these four stocks
plt.figure(figsize=(12,6))
df['Adj Close']['AAPL'].loc[start:end].rolling(window=20).mean().plot(label='20 Day Avg')
df['Adj Close']['AAPL'].loc[start:end].plot(label='BAC CLOSE')
plt.legend()

plt.figure(figsize=(12,6))
df['Adj Close']['TSLA'].loc[start:end].rolling(window=20).mean().plot(label='20 Day Avg')
df['Adj Close']['TSLA'].loc[start:end].plot(label='BAC CLOSE')
plt.legend()

plt.figure(figsize=(12,6))
df['Adj Close']['GS'].loc[start:end].rolling(window=20).mean().plot(label='20 Day Avg')
df['Adj Close']['GS'].loc[start:end].plot(label='BAC CLOSE')
plt.legend()

plt.figure(figsize=(12,6))
df['Adj Close']['MS'].loc[start:end].rolling(window=20).mean().plot(label='20 Day Avg')
df['Adj Close']['MS'].loc[start:end].plot(label='BAC CLOSE')
plt.legend()

# analyze the correlation between the stocks Close Price
sns.heatmap(df.xs(key='Adj Close',axis=1,level='Attributes').corr(),annot=True)

