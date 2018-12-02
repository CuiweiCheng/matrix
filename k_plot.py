import matplotlib.pyplot as plt
import pandas_datareader.data as wb
import pandas as pd
import pylab
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
from mpl_finance import candlestick_ohlc


def plot_k_line(stock_symbol, start='2018-01-01', end='2018-11-30'):
    """
    
    :param stock_symbol: e.g., 'AAPL'
    :param start: start date, default to '2018-01-01'
    :param end: end date, default to '2018-11-30'
    :return: a k-line graph will be plotted
    """
    # import data
    stock = wb.DataReader(stock_symbol, 'yahoo', start, end)
    # data standardization
    stock = stock.apply(pd.to_numeric, errors='coerce')
    stock.index = pd.to_datetime(stock.index, format='%Y-%m-%d')
    # plot k line
    mondays = WeekdayLocator(MONDAY)
    all_days = DayLocator()
    pylab.rcParams['figure.figsize'] = (15, 9)
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    week_format = DateFormatter('%b %d, %Y')
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(all_days)
    ax.xaxis.set_major_formatter(week_format)
    ax.grid(True)

    candlestick_ohlc(ax, list(zip(list(date2num(stock.index.tolist())), stock["Open"].tolist(), stock["High"].tolist(),
                                  stock["Low"].tolist(), stock["Close"].tolist())),
                     colorup="red", colordown="green", width=1 * .4)

    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title(stock_symbol)
    plt.show()
