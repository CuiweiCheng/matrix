import pandas as pd
import stock_price as sp
from Exploratory_Data_Analysis import EDA

from Indicator import Indicator


# df = pd.read_csv('company_list.csv')
# # print(df[:3])
# company_slice = pd.DataFrame(df, index=range(3), columns=['Symbol', 'Name', 'Sector'])
# print(company_slice)
# for i in df['Symbol'][:3]:
#     sp.download_stock_price(i)

# x = EDA(['AAPL','TSLA','GS','MS'])
# x.show_moving_avg()
# x.show_corr_map()
# x.compare_close_price()
y = Indicator('AAPL')
y.get_Volume()