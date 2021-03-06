# Group Name: The Matrix  
## Team members:  
Cuiwei Cheng / cc4309  
Jian Ji / jj2985  
Ellie Li / yl3883  
Yijie Fu / yf2474
## Class Section: IEOR E4501 Tools for Analytics Section1 
#
### All of our functions, examples and results can be viewed by simply running example.ipynb.
#
## Purpose of the project
We propose to investigate the indicators of stocks, construct and optimize a stock portfolio to yield a higher return by creating a python financial analysis library. The library consists of time series analysis, a Web Crawler, an Exploratory Data Analysis, and Model Buildings.

## Functions of our project

- [Get time series of stock prices by Web Crawler](#get-time-series-of-stock-prices-by-web-crawler)

- [Analyze stock prices by exploratory data analysis and plotting K-line graph](#analyze-stock-prices-by-exploratory-data-analysis-and-plotting-k-line-graph)

- [Investigate indicators of stock prices](#investigate-indicators-of-stock-prices)

- [Build factor models and conduct PCA to construct stock portfolios](#build-factor-models-and-conduct-pca-to-construct-stock-portfolios)

- [Build Machine Learning models, for example, Neural Network, to analyze stock portfolios by training historical data](#build-neural-network-to-analyze-stock-portfolios)

- [Optimize the portfolios by large-scale Quadratic Optimization based on Markowitz Model and Backtesting based on historical market data](#optimize-the-portfolios-by-quadratic-optimization)

## Install relevant packages before start  
To successfully achive the purpose and realize functions of our project, python packages including pandas_datareader and mpl_finance are required to be installed.  

## Get time series of stock prices by Web Crawler
### Download csv file of a stock
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
### Exploratory data analysis
* Show a line plot comparing close price of different stocks.
```python
>>> import exploratory_data_analysis as eda
>>> x = eda.EDA(['AAPL', 'TSLA', 'GS', 'MS'])
>>> x.compare_close_price()
```
![image](https://github.com/CuiweiCheng/matrix/raw/master/images/compare_price.png)
* Show the 20 day average with close price with respect to different stocks.
```python
>>> x.show_moving_avg()
```
![image](https://github.com/CuiweiCheng/matrix/raw/master/images/mov_avg.png)
* Show the heatmap of correlation between the stocks close price.
```python
>>> x.show_corr_map()
```
![image](https://github.com/CuiweiCheng/matrix/raw/master/images/heatmap.png)
### Plot K-line graph
```python
>>> import k_plot
>>> k_plot.plot_k_line('AAPL')
```
![image](https://github.com/CuiweiCheng/matrix/raw/master/images/k_line.jpg)


## Investigate indicators of stock prices
Analyze six indicators:

Indicators |Method   
---------- | -----------
**Open price**|**get_Open**
**Close price**|**get_Close**
**Volume**|**get_Volume**
**Simple Moving Average**|**get_SMA**
**Rate of Return**|**get_ROC**
**Force Index**|**get_FI**

```python
>>> import indicator
>>> x = indicator.Indicator('AAPL')
>>> x.get_Volume()
```
![image](https://github.com/CuiweiCheng/matrix/raw/master/images/get_volume.png)

## Build factor models and conduct PCA to construct stock portfolios
### Step1: Get the price of stocks
Assume we have `m` daily price of `n` stocks stored in `asset_pool_pd`, which is a `pd.DataFrame` having `n` rows * `m` columns
For example, let `asset_pool_pd` be a `pd.DataFrame` which has `947` rows x `504` columns. 
Each row represents `a stock` and each column represent `a day`.
```python
>>> import pandas as pd
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
Set the `tolerance` as the stopping condition for calculating **eigenvalue**: If the current eigenvalue is less than   
`tolerance*the max eigenvalue`, we stop calculating the `eigenvalue` because the following eigenvalue is too small and trivial.  
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
It takes 0.010636419999968894 seconds to compute Cov matrix using our own algorithm.
```
#### Step3.3: Calculate eigenvalues and eigenvectors
```python
>>> import eigen 
>>> [evalist, vlist] =eigen._estimate_spectrum(cov_matrix, tolerance)
```
### Step4: Generate lower-dimensional covariance matrix
We can now select the top `k` eigenvector to reduce dimensions.  
Example below assume `k=10`
```python
>>> import eigen 
>>> import numpy as np
>>> k=10
>>> lower_dim_mat=eigen.reduce_dimension(asset_pool_pd,vlist,k)
>>> lower_dim_mat.shape
(253, 10)
```
Therefore, the `lower_dim_mat` is now the dimension-reduced matrix from original daily price data `asset_pool_pd`, which reduce the `947` stocks to `10` principal stocks with still holding `253` days data

## Build Neural Network to analyze stock portfolios
### Step1: Set parameter for the Neural Network
We need to initialize these parameters for training and prediction:  
`n`, `m`, `t`, `gap`, `activation_func`, `epochs`, `learning_rate`, `stockdata`  
`n`: number of assets, integer  
`m`: number of neural nodes in hidden layers, integer  
`t` : set the first `t` days return data of `n` assets as the traing dataset
`gap` : we use data of day `t` to predict data of day `t+gap`, integer
`activation_func`: type of activation function, string  
(We define three types of activation function in this module: `'tanh'`, `'relu'`, `'sigmoid'`)  
`epochs`: number of iterations in training  
`learning_rate`: stride in backpropogation affecting the update amount of parameters (vector 'W' and 'b' on each arc) in Neural Network 
`stockdata`: `n` rows (assets) and several colomns (days) 

```python
>>> import NeuralNetwork 
>>> n=44; m=50; epochs=1000; gap=3; activation_func='tanh'; learning_rate=0.0000001; stockdata=asset_pool_return_pd; t=10
```

### Step2: Predict stock price
In this step, we first train the Neural Network by using the first `0 to t` days of return data.   
We use data on day `i` to train data on day `i+gap` and then utilize the trained NN model to predict the second part `t to end` days of return.   
The output data `Y_hat` is the prediction value of the return on day `t to end` and the `total_cost` (sum square of error of each day and each asset).
```python
>>> import NeuralNetwork 
>>> Y_hat, total_cost=NeuralNetwork.NNPredict(n,m,t, gap, activation_func, epochs, learning_rate,stockdata)
...
i =  997 cost =  7.101546276402318 avg_unit_cost 2.975790833376208e-05
i =  998 cost =  7.101546943819563 avg_unit_cost 2.975791113046866e-05
i =  999 cost =  7.101547612308513 avg_unit_cost 2.975791393166605e-05
>>> Y_hat.shape
(44, 239)
```
Therefore, `Y_hat` has prediction return of `44` asset on `239` days

## Optimize the portfolios by Quadratic Optimization
### Step1: Set parameters
Set the  `lower bounds`, `upper bounds` and `mu`(which is expected return rate for `n` stocks.   
```python   
>>> import random
>>> import numpy as np
>>> import pandas as pd
>>> covariance=pd.DataFrame(cov_matrix)
>>> n=len(covariance)
>>> lower=np.array([random.uniform(0, 0.05) for i in range(n)])
>>> upper=np.array([random.uniform(0.9, 1) for i in range(n)])
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
Set `lambda` denoting the risk preference in the Markowitz Model, larger `lambda` means **risk preference** 
```python   
>>> lam=10
```
Set (or calculated from original stock price data) the covariance matrix of the `n` stocks 
```python   
>>> import pandas as pd
>>> covariance=pd.DataFrame(cov_matrix)
```
### Step2: Utilize `First Order Method` and iterate enough times to achieve optimal weight of `n` stocks
```python
>>> from quadratic import quadratic_opt 
>>> n=len(covariance)
>>> matrix=np.vstack((lower,upper,mu)).T
>>> matrix=pd.DataFrame(data=matrix,columns=['lower', 'upper', 'mu'])
>>> result = quadratic_opt(n,lam,matrix,covariance)
>>> if type(result)!=str:
        x_new = result[0]
        F_new = result[1]
    else:
        print('Output message is', result)
```
We then get the optimal weight of `n` stocks and the corresponding minimum value of the Markowitz's Objective Function. Therefore, since we have the optimal weight of the `n` stocks, we can base on this weight to construct the portfolio and conduct further analysis.

**One small-scale example (4 assets):**
```python
>>> from quadratic import quadratic_opt 
>>> n=4
>>> lam=10
>>> matrix
         lower  upper     mu
asset_1  0.010    0.5  20.00
asset_2  0.000    1.0   0.04
asset_3  0.005    1.0   0.10
asset_4  0.030    0.4  -0.05
>>> covariance
         asset_1  asset_2  asset_3  asset_4
asset_1    54.00     -0.3    -0.02     0.00
asset_2    -0.30     12.0     0.50     1.00
asset_3    -0.02      0.5     4.00     0.30
asset_4     0.00      1.0     0.30     0.02
>>> x_new, F_new = quadratic_opt(n,lam,matrix,covariance)
...
x= [0.05486039 0.10914633 0.43694331 0.39904997] F= 11.946741691325817
x= [0.0547931  0.10986011 0.4362954  0.39905139] F= 11.946661522359891
x= [0.0545109  0.10916799 0.43726374 0.39905737] F= 11.946527345603972
>>> x_new
array([0.0545109 , 0.10916799, 0.43726374, 0.39905737])
```
**Further backtesting**  
We can now calculating the historic value and maximum drawdown of the portfolio based on these weights of `n` assets.  
In the module `quadratic`, we provide the method `backtest` and `max_drawdown` to calculate and plot.   
Above example continues:
```python
>>> from quadratic import backtest, max_drawdown
>>> p_mat=asset_pool_pd[0:4]
>>> p_mat
array([[ 5.31  ,  5.14  ,  5.14  , ...,  7.35  ,  7.85  ,  8.2   ],
       [ 1.33  ,  1.3385,  1.3385, ...,  1.4504,  1.49  ,  1.5   ],
       [74.22  , 75.58  , 73.99  , ..., 33.4   , 33.15  , 33.3   ],
       [ 6.16  ,  5.95  ,  5.15  , ...,  7.93  ,  8.1   ,  8.28  ]])
```
```python
>>> value_port=backtest(x_new,p_mat)
```
![image](https://github.com/CuiweiCheng/matrix/raw/master/images/backtest_demo.png)
```python
>>> value_port
array([35.34655418, 35.84909189, 34.83459665, 33.84750544, 32.47909014,
       32.57250577, 32.98604229, 32.79489458, 33.06224579, 34.20150541,
       ...
       17.7890263 , 18.35813423, 17.98212881, 17.6029267 , 17.76566572,
       18.3281261 , 18.31822843, 18.47581881])
>>> max_drawdown(value_port)
0.5217494113999498
```
**Another example: Combine Quadratic Optimization with PCA to construct portfolio**  
We now have the `lower_dim_mat` generated from PCA. Then we can apply the quadratic optimization on this new **virtual daily price data**
```python   
>>> import random
>>> import eigen 
>>> import numpy as np
>>> import pandas as pd
>>> from quadratic import quadratic_opt 

>>> asset_pool_return_pd=eigen.calculate_return_rate(lower_dim_mat)
>>> cov_matrix=eigen.calculate_cov(asset_pool_return_pd)
>>> covariance=pd.DataFrame(cov_matrix)
>>> n=len(covariance)
>>> lower=np.array([random.uniform(0, 0.1) for i in range(n)])
>>> upper=np.array([random.uniform(0.9, 1) for i in range(n)])
>>> mu=asset_pool_return_pd.mean(axis=1)
>>> mu=mu.values

>>> matrix=np.vstack((lower,upper,mu)).T
>>> matrix=pd.DataFrame(data=matrix,columns=['lower', 'upper', 'mu'])
>>> result = quadratic_opt(n,lam,matrix,covariance)
>>> if type(result)!=str:
        x_new = result[0]
        F_new = result[1]
    else:
        print('Output message is', result)
```
Therefore, now `x_new` is the optimal vector of weights for `lower_dim_mat` with `k-dimension` 
