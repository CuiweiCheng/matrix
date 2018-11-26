import pandas as pd
import stock_price as sp


df = pd.read_csv('company_list.csv')
# print(df[:3])
company_slice = pd.DataFrame(df, index=range(3), columns=['Symbol', 'Name', 'Sector'])
print(company_slice)
for i in df['Symbol'][:3]:
    sp.download_stock_price(i)