import requests
from lxml import etree
import pandas as pd
import os


def download_stock_price(stock_symbol):

    """
    enter a string of stock symbol, a csv file will be downloaded automatically into the current path
    """

    url = f'https://charting.nasdaq.com/ext/charts.dll?2-1-14-0-0-512-03NA000000\
{stock_symbol}\
-&SF:6|8|5-SH:8=20-BG=FFFFFF-BT=0-WD=605-HT=395--xtbl'

    r = requests.get(url)
    html = etree.HTML(r.text)

    result = html.xpath('/html/body/center[1]/table/tr[1]/td[2]/font/a/@href')

    a = result[0].encode('utf-8')
    b = str(a)

    download_url = 'https://charting.nasdaq.com/ext/' + b[2:-4] + 'xcl-'
    # print(download_url)
    r = requests.get(download_url)
    with open(f"{stock_symbol}.csv", "wb") as code:
        code.write(r.content)
    # print(f'stock price of {stock_symbol} downloaded, you can find it in the current path')


def dataframe_of_single_stock(stock_symbol):

    """
    enter a string of a stock symbol, a dataframe will be returned.
    This dataframe contains columns Date, Open(high, low, close) Price, Volume, etc.
    """

    download_stock_price(stock_symbol)
    x = pd.read_csv(f'{stock_symbol}.csv', sep='\t')
    os.remove(f'{stock_symbol}.csv')
    return x


def dataframe_of_stocks(list_of_stock_symbol):

    """
    this function takes a list of stock symbols, and returns a dataframe.
    indexes are different date, columns are different stocks.
    """

    combine_price = pd.DataFrame()
    for i in list_of_stock_symbol:
        download_stock_price(i)
        df = pd.read_csv(f'{i}.csv', sep='\t')
        try:
            if len(df['Close/Last']) < 250:
                print(f'No enough trading day for stock {i}')
            else:
                combine_price.insert(loc=0, column=f'{i}', value=df['Close/Last'])
                print(f'stock {i} has been loaded just now')
        except KeyError:
            print(f'No data available for stock {i}')
        os.remove(f'{i}.csv')
    combine_price.index = range(1, len(combine_price) + 1)
    return combine_price
