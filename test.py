import pandas as pd
import numpy as np

df = pd.read_csv('chh.csv', sep='\t')
# print(df)
# print(df.columns)
print(df['Date'][0])
# print(type(df['Date']))
