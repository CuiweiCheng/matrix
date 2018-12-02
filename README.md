# Matrix
We propose to investigate the indicators of stocks, construct and optimize a stock portfolio to yield a higher return by creating a python financial analysis library. The library consists of time series analysis, a Web Crawler, an Exploratory Data Analysis, and Model Buildings.

## Functions of our project

- [Get time series of stock prices by Web Crawler](#get-time-series-of-stock-prices-by-web-crawler)

- Analyze stock prices by exploratory data analysis and plotting K-line graph

- Investigate indicators of stock prices

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

## Build factor models and conduct PCA to construct stock portfolios
### Step1: Get the price of stocks
Assume we have `m` daily price of `n` stocks stored in `asset_pool_pd`, which is a `pd.DataFrame` having `n` rows * `m` columns
For example, let `asset_pool_pd` be a `pd.DataFrame` which has `947` rows x `504` columns. 
Each row represents `a stock` and each column represent `a day`.
```python
>>> asset_pool_pd.head()
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
```
We can take an insight into the module `eigen` and function `calculate_eigens`  
```python
>>> import eigen
>>> dir(eigen)
```
It includes the following three steps:  
1. Calculate the return rate matrix
2. Calculate covariance matrix of rate of return  
3. Calculate eigenvalues and eigenvectors  
#### Step3.1 Calculate rate of return
```python
>>> import eigen 
>>> asset_pool_return_pd=eigen.calculate_return_rate(asset_pool_pd)
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

>>> mu=mu.value
```
### Step2: Utilize `First Order Method` and iterate enough times to achieve optimal weight of `n` stocks


We then get the optimal weight of `n` stocks and the corresponding minimum value of the portfolio based on these `n` stocks  

