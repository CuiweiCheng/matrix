
import matplotlib.pyplot as plt
import pandas_datareader.data as wb
import pandas as pd
import numpy as np
import datetime


# import data
start = '2018-01-01'
end = '2018-11-30'

aapl = wb.DataReader('AAPL', 'yahoo', start, end)
tsla = wb.DataReader('TSLA', 'yahoo', start, end)
gs = wb.DataReader('GS', 'yahoo', start, end)
ms = wb.DataReader('MS', 'yahoo', start, end)

# data standarization
aapl.head()
aapl = aapl.apply(pd.to_numeric, errors='coerce') 
aapl.index = pd.to_datetime(aapl.index, format='%Y-%m-%d') 

tsla.head()
tsla = tsla.apply(pd.to_numeric, errors='coerce') 
tsla.index = pd.to_datetime(aapl.index, format='%Y-%m-%d') 

gs.head()
gs = gs.apply(pd.to_numeric, errors='coerce') 
gs.index = pd.to_datetime(aapl.index, format='%Y-%m-%d') 

ms.head()
ms = ms.apply(pd.to_numeric, errors='coerce') 
ms.index = pd.to_datetime(aapl.index, format='%Y-%m-%d') 

# plot k-line
    # download a matplotlib.finance from github
'''
git clone https://github.com/matplotlib/mpl_finance.git

cd mpl_finance/

python setup.py build

python setup.py install

from matplotlib.finance import candlestick_ochl

--> from mpl_finance import candlestick_ochl

'''


from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
from mpl_finance import candlestick_ohlc, candlestick2_ochl
import datetime
from datetime import date


# K-plot for APPLE
mondays = WeekdayLocator(MONDAY)        
alldays = DayLocator()    

%pylab inline
pylab.rcParams['figure.figsize'] = (15, 9)      
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
weekFormatter = DateFormatter('%b %d, %Y')  
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(weekFormatter)
ax.grid(True)

candlestick_ohlc(ax, list(zip(list(date2num(aapl.index.tolist())), aapl["Open"].tolist(), aapl["High"].tolist(),
                 aapl["Low"].tolist(), aapl["Close"].tolist())),
                 colorup = "red", colordown = "green", width = 1 * .4)

ax.autoscale_view()
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('AAPL')
plt.show()


# K-plot for Tesla

mondays = WeekdayLocator(MONDAY)        
alldays = DayLocator()    

%pylab inline
pylab.rcParams['figure.figsize'] = (15, 9)      
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
weekFormatter = DateFormatter('%b %d, %Y')  
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(weekFormatter)
ax.grid(True)

candlestick_ohlc(ax, list(zip(list(date2num(tsla.index.tolist())), tsla["Open"].tolist(), tsla["High"].tolist(),
                 tsla["Low"].tolist(), tsla["Close"].tolist())),
                 colorup = "red", colordown = "green", width = 1 * .4)

ax.autoscale_view()
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('TSLA')
plt.show()

# K-plot for Microsoft

mondays = WeekdayLocator(MONDAY)        
alldays = DayLocator()    

%pylab inline
pylab.rcParams['figure.figsize'] = (15, 9)      
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
weekFormatter = DateFormatter('%b %d, %Y')  
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(weekFormatter)
ax.grid(True)

candlestick_ohlc(ax, list(zip(list(date2num(ms.index.tolist())), ms["Open"].tolist(), ms["High"].tolist(),
                 ms["Low"].tolist(), ms["Close"].tolist())),
                 colorup = "red", colordown = "green", width = 1 * .4)

ax.autoscale_view()
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('MS')
plt.show()


# K-plot for Goldman Sachs

mondays = WeekdayLocator(MONDAY)        
alldays = DayLocator()    

%pylab inline
pylab.rcParams['figure.figsize'] = (15, 9)      
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
weekFormatter = DateFormatter('%b %d, %Y')  
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(weekFormatter)
ax.grid(True)

candlestick_ohlc(ax, list(zip(list(date2num(gs.index.tolist())), gs["Open"].tolist(), gs["High"].tolist(),
                 gs["Low"].tolist(), gs["Close"].tolist())),
                 colorup = "red", colordown = "green", width = 1 * .4)

ax.autoscale_view()
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('GS')
plt.show()


