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
### download stock price
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

## Build factor models and conduct PCA to construct stock portfolios
### Step1: Get the price of stocks

### Step2: Calculate rate of return

### Step3: Calculate the covariance matrix of rate of return

### Step4: Reduce dimensions by PCA

### Step5: Generate lower-dimensional covariance matrix

```python
>>> import eigen
>>> dir(eigen)
>>> asset_pool_pd=eigen.calculate_return_rate(asset_pool_pd)
>>> asset_pool_pd=eigen.calculate_cov(asset_pool_pd)
>>> evalist,vlist=eigen.calculate_eigens(asset_pool_pd)
```

## Build Neural Network to analyze stock portfolios
### Step1: Set parameter for the Neural Network

### Step2: Train the Neural Network

### Step3: Predict stock price

## Optimize the portfolios by Quadratic Optimization
### Step1: Set parameters
Set the lower bounds, upper bounds and expected return rate for n stocks.   
Set lambda denoting the risk preference in the Markov Model:   
Set (or calculated from original stock price data) the covariance matrix of the n stocks  
### Step2: Utilize First Order Method and iterate enough times
We then get the optimal weight of n stocks and the corresponding minimum value of the portfolio based on these n stocks  

