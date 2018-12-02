import matplotlib.pyplot as plt
import pandas_datareader.data as wb


class Indicator:
    """
    This class is used to find the indicator of individual stock
    """

    def __init__(self, symbol, start='2017-01-01', end='2018-01-01'):
        self.symbol = symbol
        self.start = start
        self.end = end

        self.df = wb.DataReader(self.symbol, 'yahoo', self.start, self.end)

    def get_SMA(self, windows=20):
        """

        :param windows: default to 20
        :return: a graph of simple moving average
        """
        sma = self.df.rolling(window=windows).mean()['Adj Close']

        ax = sma.plot(title=self.symbol)
        ax.set_xlabel('date')
        ax.set_ylabel('Simple Moving Average')
        ax.grid()
        plt.show()

        return sma

    def get_ROC(self):
        """

        :return: a graph of return of change
        """
        pct = self.df.pct_change()['Adj Close']

        ax = pct.plot(title=self.symbol)
        ax.set_xlabel('date')
        ax.set_ylabel('Return of Change')
        ax.grid()
        plt.show()

        return pct

    def get_ForceIndex(self):
        """

        :return: a graph of force index
        """
        FI = (self.df['Close'] - self.df['Open']) * self.df['Volume']
        ax = FI.plot(title=self.symbol)
        ax.set_xlabel('date')
        ax.set_ylabel('Force Index')
        ax.grid()
        plt.show()

        return FI

    def get_Open(self):
        """

        :return: a graph of open price
        """
        Open = self.df['Open']
        ax = Open.plot(title=self.symbol)
        ax.set_xlabel('date')
        ax.set_ylabel('open price')
        ax.grid()
        plt.show()

        return self.df['Open']

    def get_Close(self):
        """

        :return: a graph of close price
        """
        close = self.df['Close']
        ax = close.plot(title=self.symbol)
        ax.set_xlabel('date')
        ax.set_ylabel('close price')
        ax.grid()
        plt.show()

        return self.df['Close']

    def get_Volume(self):
        """

        :return: a graph of volume
        """
        vol = self.df['Volume']
        ax = vol.plot(title=self.symbol)
        ax.set_xlabel('date')
        ax.set_ylabel('Volume')
        ax.grid()
        plt.show()

        return vol