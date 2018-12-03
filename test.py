import pandas as pd
import stock_price as sp
import exploratory_data_analysis as eda
import indicator

# # test for web crawler
# sp.download_stock_price('AAPL')
#
# x = sp.dataframe_of_single_stock('TSLA')
# print(x)
#
# y = sp.dataframe_of_stocks(['BIDU', 'SINA'])
# print(y)
#
# df = pd.read_csv('company_list.csv')
# list_of_stock_symbol = df['Symbol'][:5]  # first twenty stocks in the company list provided.
# z = sp.dataframe_of_stocks(list_of_stock_symbol)
# print(z)

# # test for EDA
# x = eda.EDA(['AAPL', 'TSLA', 'GS', 'MS'])
# x.compare_close_price()
# x.show_moving_avg()
# x.show_corr_map()

# test for indicator
x = indicator.Indicator('AAPL')
x.get_Volume()