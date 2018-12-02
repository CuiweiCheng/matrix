'''
1. open price/ close price/ volumn
2. Simple Moving Average
3. Rate of Return
4. Force Index 
'''

import matplotlib.pyplot as plt
import pandas_datareader.data as wb
import pandas as pd
import numpy as np
import datetime
import seaborn as sns



### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
### First analyze the indicators of stock simultaneously
### Then create a class to figure out the indicator for indivdual stock
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

### we analyze the indicators of stock simultaneously first
### 
### 

# import data
start = '2016-01-01'
end = '2018-11-30'

aapl = wb.DataReader('AAPL', 'yahoo', start, end)
tsla = wb.DataReader('TSLA', 'yahoo', start, end)
gs = wb.DataReader('GS', 'yahoo', start, end)
ms = wb.DataReader('MS', 'yahoo', start, end)

df = wb.DataReader(['AAPL','TSLA','GS','MS'], 'yahoo', start, end)
df.info()
df.describe()

tickers = ['AAPL', 'TSLA', 'MS', 'GS']

# 1.1 create a line plot showing Open price for these four stocks

for tick in tickers:
    df['Open'][tick].plot(figsize=(12,4),label=tick)
plt.legend()

# analyze the correlation between the stocks open price
sns.heatmap(df.xs(key='Open',axis=1,level='Attributes').corr(),annot=True)

sns.clustermap(df.xs(key='Open',axis=1,level='Attributes').corr(),annot=True)


# 1.2 create a line plot showing Close price for these four stocks

for tick in tickers:
    df['Close'][tick].plot(figsize=(12,4),label=tick)
plt.legend()

# analyze the correlation between the stocks close price
sns.heatmap(df.xs(key='Close',axis=1,level='Attributes').corr(),annot=True)

sns.clustermap(df.xs(key='Close',axis=1,level='Attributes').corr(),annot=True)

# 1.3 create a line plot showing Volume for these four stocks

for tick in tickers:
    df['Volume'][tick].plot(figsize=(12,4),label=tick)
plt.legend()

# analyze the correlation between the stocks volume
sns.heatmap(df.xs(key='Volume',axis=1,level='Attributes').corr(),annot=True)

sns.clustermap(df.xs(key='Volume',axis=1,level='Attributes').corr(),annot=True)




# 2. create a line plot showing 20 days SMA  for these four stocks


for tick in tickers:
    df['ROC',tick] = df.rolling(window=20).mean()['Close'][tick]

    df['ROC',tick].plot(figsize=(12,4),label=tick)
    
plt.legend()

# analyze the correlation between the stocks SMA
sns.heatmap(df.xs(key='ROC',axis=1,level='Attributes').corr(),annot=True)

sns.clustermap(df.xs(key='ROC',axis=1,level='Attributes').corr(),annot=True)



# 5. create a line plot showing return of rate for these four stocks


for tick in tickers:
    df['pct',tick] = df['Close'][tick].pct_change()
    df['pct',tick].plot(figsize=(12,4),label=tick)
    
plt.legend()


# analyze the correlation between the stocks PCT
sns.heatmap(df.xs(key='pct',axis=1,level='Attributes').corr(),annot=True)

sns.clustermap(df.xs(key='pct',axis=1,level='Attributes').corr(),annot=True)


### This class is used to find the indicator of individual stock
### 
### 

class Indicator:
    def __init__(self, symbol, start, end):
        self.symbol = symbol
        self.start = start
        self.end = end
    
    def get_data(self):
        df = wb.DataReader(self.symbol, 'yahoo', start, end)
        self.df = df
        
        close = self.df['Adj Close']
        self.df.fillna(0)
        
        ax = close.plot(title=self.symbol)
        ax.set_xlabel('date')
        ax.set_ylabel('Stock Price')
        ax.grid()
        plt.show()
        
        return self.df.info()
    
    def get_SMA(self, windows):
        sma = self.df.rolling(window=windows).mean()['Adj Close']
        
        ax = sma.plot(title=self.symbol)
        ax.set_xlabel('date')
        ax.set_ylabel('Simple Moving Average')
        ax.grid()
        plt.show()
        
        return sma
    
    def get_ROC(self):
        pct = self.df.pct_change()['Adj Close'] 
        
        ax = pct.plot(title=self.symbol)
        ax.set_xlabel('date')
        ax.set_ylabel('Return of Change')
        ax.grid()
        plt.show()
        
        return pct
    
    def get_ForceIndex(self):
        FI = (self.df['Close']-self.df['Open'])*self.df['Volume']
        
        ax = FI.plot(title=self.symbol)
        ax.set_xlabel('date')
        ax.set_ylabel('Force Index')
        ax.grid()
        plt.show()
        
        return FI
    
    def get_Open(self):
        Open = self.df['Open']
        ax = Open.plot(title=self.symbol)
        ax.set_xlabel('date')
        ax.set_ylabel('open price')
        ax.grid()
        plt.show()
        
        return self.df['Open']
    
    def get_Close(self):
        close = self.df['Close']
        ax = close.plot(title=self.symbol)
        ax.set_xlabel('date')
        ax.set_ylabel('close price')
        ax.grid()
        plt.show()
        
        return self.df['Close']
    
    def get_Volume(self):
        vol = self.df['Volume']
        ax = vol.plot(title=self.symbol)
        ax.set_xlabel('date')
        ax.set_ylabel('Volume')
        ax.grid()
        plt.show()
        
        return vol
        


start = '2017-01-01'
end = '2018-01-01'
indicator = Indicator('AAPL', start, end)
indicator.get_data()
indicator.get_SMA(20)
indicator.get_ROC()
indicator.get_Open()
indicator.get_Close()
indicator.get_ForceIndex()
indicator.get_Volume()






