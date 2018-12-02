# Matrix
We propose to investigate the indicators of stocks, construct and optimize a stock portfolio to yield a higher return by creating a python financial analysis library. The library consists of time series analysis, a Web Crawler, an Exploratory Data Analysis, and Model Buildings.

## Functions of our project

- [Get time series of stock prices by Web Crawler](#get-time-series-of-stock-prices-by-web-crawler)

- [Analyze stock prices by exploratory data analysis and plotting K-line graph](#analyze-stock-prices-by-exploratory-data-analysis-and-plotting-k-line-graph)

- [Investigate indicators of stock prices](#investigate-indicators-of-stock-prices)

- [Build factor models and conduct PCA to construct stock portfolios](#build-factor-models-and-conduct-pca-to-construct-stock-portfolios)

- [Build Machine Learning models, for example, Neural Network, to analyze stock portfolios by training historical data](#build-neural-network-to-analyze-stock-portfolios)

- [Optimize the portfolios by large-scale Quadratic Optimization based on Markowitz Model and Backtesting based on historical market data](#optimize-the-portfolios-by-quadratic-optimization)

## Get time series of stock prices by Web Crawler
### Download cvs file of a stock
Enter a string of stock symbol, a csv file will be downloaded automatically into the current path

Company / Index|Symbol                                                                                                                                             
---------- | -----------
**Apple Inc.**|**AAPL**
**Tesla, Inc.**|**TSLA**
**Dow Jones Industrial Average** |**DJIA**                                                 
**Standard & Poor's 500**| **SPX**
```python
>>> import stock_price as sp
>>> sp.download_stock_price('aapl')
```
### Get a dataframe of a single stock
Enter a string of a stock symbol, a dataframe will be returned. This dataframe contains columns Date, Open(high, low, close) Price, Volume, etc.
```python
>>> import stock_price as sp
>>> x = sp.dataframe_of_single_stock('TSLA')
>>> print(x)
```
### Get a dataframe of several stocks
This function takes a list of stock symbols, and returns a dataframe. Indexes are different date, columns are different stocks.
```python
>>> import stock_price as sp
>>> y = sp.dataframe_of_stocks(['BIDU', 'SINA'])
>>> print(y)
```
## Analyze stock prices by exploratory data analysis and plotting K-line graph
### Step1: Get the price of stocks
Analysze Apple, Tesla, Goldman Sachs and Microsoft
#### Step1.1: Import data
```python
>>>import pandas_datareader.data as wb
>>>aapl = wb.DataReader('AAPL', 'yahoo', start, end)
>>>df = wb.DataReader(['AAPL','TSLA','GS','MS'], 'yahoo', start, end)
>>>df.info()
>>>df.describe()
```
#### Step1.2: Get a basic description
```python
>>>df.info()
>>>df.describe()
```

#### Step1.3: Deal with missing value
```python
>>>aapl.interpolate()
```
### Step2: Visualize time series of data
```python
>>>aapl['Close'].plot(grid=True)
>>>plt.show()
```
#### Step2.1: Plot the closing prices and simple moving average
```python
>>>tickers = ['AAPL', 'TSLA', 'MS', 'GS']
>>>for tick in tickers:
>>>    df['Adj Close'][tick].plot(figsize=(12,4),label=tick)
>>>plt.legend()
```
#### Step2.2: Investigate the correlation
```python
>>>import seaborn as sns
>>>sns.heatmap(df.xs(key='Adj Close',axis=1,level='Attributes').corr(),annot=True)
```
### Step3: Plot K-line graph
download a matplotlib.finance

git clone https://github.com/matplotlib/mpl_finance.git
cd mpl_finance/
python setup.py build
python setup.py install
from matplotlib.finance import candlestick_ochl
--> from mpl_finance import candlestick_ochl


## Investigate indicators of stock prices
### Step1: Get the price of stocks

### Step2: Financial Analyses
First analyze the indicators of stocks simultaneously. Investigate the correlations between different indicators
Also, create a class to figure out the indicator for indivdual stock

Analyze six indicators:

Indicators |Method   
---------- | -----------
**Open price**|**get_Open**
**Close price**|**get_Close**
**Volume**|**get_Volume**
**Simple Moving Average**|**get_SMA**
**Rate of Return**|**get_ROC**
**Force Index**|**get_FI**


## Build factor models and conduct PCA to construct stock portfolios
### Step1: Get the price of stocks
Assume we have `m` daily price of `n` stocks stored in `asset_pool_pd`, which is a `pd.DataFrame` having `n` rows * `m` columns
For example, let `asset_pool_pd` be a `pd.DataFrame` which has `947` rows x `504` columns. 
Each row represents `a stock` and each column represent `a day`.
```python
>>> df = pd.read_csv('company_list.csv')
>>> list_of_stock_symbol = df['Symbol'][:50]  # first fifty stocks in the company list provided.
>>> asset_pool_pd = sp.dataframe_of_stocks(list_of_stock_symbol)
>>> asset_pool_pd=asset_pool_pd.T
>>> asset_pool_pd.head()
        0        1        2      3    ...      249      250    251    252
IOTS   5.31   5.1400   5.1400   5.27  ...     7.55   7.3500   7.85   8.20
AEY    1.33   1.3385   1.3385   1.35  ...     1.48   1.4504   1.49   1.50
ADUS  74.22  75.5800  73.9900  71.75  ...    32.20  33.4000  33.15  33.30
ADAP   6.16   5.9500   5.1500   5.11  ...     7.80   7.9300   8.10   8.28
ADMP   3.05   2.7700   2.7500   2.65  ...     4.75   4.0000   4.05   3.80

[5 rows x 253 columns]
```

### Step2: Set the parameter `tolerance`
Set the `tolerance` as the stopping condition for calculating **eigenvalue**: If the current eigenvalue is less than `tolerance*the max eigenvalue`, we stop calculating the `eigenvalue` because the following eigenvalue is too small and trivial.
For example:  
```python
>>> tolerance=0.0000001
```
### Step3: Generate eigenvalue and eigenvector
```python
>>> from eigen import calculate_eigens 
>>> evalist,vlist=calculate_eigens(asset_pool_pd,tolerance)
>>> len(evalist)
44
>>> len(vlist)
44
>>> len(vlist[0])
44
```
We can take an insight into the module `eigen` and function `calculate_eigens`  
```python
>>> import eigen
>>> dir(eigen)
['__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__spec__',
 '__warningregistry__',
 '_eigenvalue',
 '_estimate_spectrum',
 'calculate_cov',
 'calculate_eigens',
 'calculate_return_rate',
 'np',
 'pd',
 'sys',
 'time']
```
It includes the following three steps:  
1. Calculate the return rate matrix
2. Calculate covariance matrix of rate of return  
3. Calculate eigenvalues and eigenvectors  
#### Step3.1 Calculate rate of return
```python
>>> import eigen 
>>> asset_pool_return_pd=eigen.calculate_return_rate(asset_pool_pd)
>>> asset_pool_return_pd.head()
           1         2         3      ...          250       251       252
IOTS -0.032015  0.000000  0.025292    ...    -0.026490  0.068027  0.044586
AEY   0.006391  0.000000  0.008592    ...    -0.020000  0.027303  0.006711
ADUS  0.018324 -0.021037 -0.030274    ...     0.037267 -0.007485  0.004525
ADAP -0.034091 -0.134454 -0.007767    ...     0.016667  0.021438  0.022222
ADMP -0.091803 -0.007220 -0.036364    ...    -0.157895  0.012500 -0.061728

[5 rows x 252 columns]
```
#### Step3.2: Calculate the covariance matrix of rate of return
```python
>>> import eigen 
>>> cov_matrix=eigen.calculate_cov(asset_pool_return_pd)
```
#### Step3.3: Calculate eigenvalues and eigenvectors
```python
>>> import eigen 
>>> [evalist, vlist] =eigen._estimate_spectrum(cov_matrix, tolerance)
```
### Step4: Generate lower-dimensional covariance matrix
```python
>>> import eigen 
>>> import numpy as np
>>> lower_dim_mat=cov_matrix*np.array(vlist)
>>> len(lower_dim_mat)
44
>>> len(lower_dim_mat[0])
44
```

## Build Neural Network to analyze stock portfolios
### Step1: Set parameter for the Neural Network
We need to initialize these parameters for training and prediction:  
`n`, `m`, `t`, `activation_func`, `epochs`, `learning_rate`, `stockdata`  
`n`: number of assets, integer  
`m`: number of neural nodes in hidden layers, integer  
`activation_func`: type of activation function, string  
(We define three types of activation function in this module: `'tanh'`, `'relu'`, `'sigmoid'`)  
`epochs`: number of iterations in training  
`learning_rate`: stride in backpropogation affecting the update amount of parameters (vector 'W' and 'b' on each arc) in Neural Network  
`stockdata`: n rows and 

```python
>>> import NeuralNetwork 
>>> n=947; m=50; epochs=1000, learning_rate=0.0000001, stockdata=asset_pool_return_pd
```

### Step2: Predict stock price
In this step, we first train the Neural Network by using the first half days of return data and then predict the second half days of return given the `n` return data on day `t` as input data. The output data is the prediction value of the return.
```python
>>> import NeuralNetwork 
>>> Y_hat, total_cost=NeuralNetwork.NNPredict(n,m,t, "relu", epochs, learning_rate,stockdata)
```
## Optimize the portfolios by Quadratic Optimization
### Step1: Set parameters
Set the  `lower bounds`, `upper bounds` and `mu`(which is expected return rate for `n` stocks.   
Set `lambda` denoting the risk preference in the Markowitz Model:   
Set (or calculated from original stock price data) the covariance matrix of the `n` stocks 

```python
>>> import quadratic_opt from quadratic   
>>> mu=asset_pool_return_pd.mean(axis=1)
>>> mu.head()
IOTS    0.002731
AEY     0.000636
ADUS   -0.002932
ADAP    0.002174
ADMP    0.002430
dtype: float64
>>> mu=mu.values
```
### Step2: Utilize `First Order Method` and iterate enough times to achieve optimal weight of `n` stocks
```python
>>> import quadratic_opt from quadratic   
>>> quadratic_opt(n,lam,matrix,covariance)
```
We then get the optimal weight of `n` stocks and the corresponding minimum value of the Markowitz's Objective Function these weights of `n` stocks. Therefore, since we have the optimal weight of the `n` stocks, we can base on this weight to construct the portfolio and conduct further analysis.

