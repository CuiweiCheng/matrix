import pandas as pd
import stock_price as sp

# sp.download_stock_price('AAPL')
#
x = sp.dataframe_of_single_stock('TSLA')
print(x)
#
# y = sp.dataframe_of_stocks(['BIDU', 'SINA'])
# print(y)

# for teammates:
# if you guys want to create a dataframe with lots of stocks, see this.
df = pd.read_csv('company_list.csv')
list_of_stock_symbol = df['Symbol'][:50]  # first twenty stocks in the company list provided.
z = sp.dataframe_of_stocks(list_of_stock_symbol)
print(z)