import matplotlib.pyplot as plt
import pandas_datareader.data as wb


class EDA(object):
    """
    This is exploratory data analysis for stock prices.
    It should be initialized with a stock list, e.g., ['AAPL','TSLA','GS','MS']
    It can also take start date and end date, which are default to start='2016-01-01', end='2018-11-30'
    """
    def __init__(self, stock_list, start='2016-01-01', end='2018-11-30'):
        self.stock_list = stock_list
        self.start = start
        self.end = end
        self.df = wb.DataReader(self.stock_list, 'yahoo', self.start, self.end)

    def compare_close_price(self):
        """
        create a line plot showing Close price for these stocks
        :return: show a line plot comparing close price of different stocks.
        """
        for i in self.stock_list:
            self.df['Close'][i].plot(figsize=(12, 4), label=i)
            plt.legend()
        plt.show()

    def show_moving_avg(self):
        """
        create n graphs.
        :return: show the 20 day average with close price with respect to different stocks.
        """
        for i in self.stock_list:
            plt.figure(figsize=(12, 6))
            self.df['Adj Close'][i].loc[self.start:self.end].rolling(window=20).mean().plot(label=f'{i} 20 Day Avg')
            self.df['Adj Close'][i].loc[self.start:self.end].plot(label=f'{i} CLOSE')
            plt.legend()
        plt.show()

    def show_corr_map(self):
        """
        analyze the correlation between the stocks Close Price
        :return: show the heatmap of correlation between the stocks close price
        """
        import seaborn as sns
        sns.heatmap(self.df.xs(key='Adj Close', axis=1, level='Attributes').corr(), annot=True)
        # sns.clustermap(self.df.xs(cockey='Adj Close', axis=1, level='Attributes').corr(), annot=True)
        plt.show()