import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

# df = pd.read_parquet('new_changes_done.parquet')
# stocks = df['stock_id'].unique()
# dates = df['report_date'].unique()
#
# result = pd.DataFrame()
# i = 0
# for stock in stocks:
#     selected = df.loc[(df['stock_id'] == stock)]
#     selected['last_holding'] = selected['is_holding'].shift(1)
#     result = result.append(selected)
#     i = i+1
#     print(i, stock, "Done")
#
#
# #
# # result = df
# result.loc[(result['last_holding'] == 0) & (result['is_holding'] == 1), 'new_changes'] = 1
# result.loc[(result['last_holding'].isna()), 'new_changes'] = 1
# result.loc[(result['last_holding'] == 1) & (result['is_holding'] == 0), 'new_changes'] = 1
# # result['new_changes'] = result['new_changes'].fillna(0)


df = pd.read_parquet('new_changes_done.parquet')
dates = df['report_date'].unique()
# print(dates[0])
df.loc[df['report_date'] == dates[0],'new_changes'] = 0
# print(df.loc[df['report_date'] == dates[0]][['report_date','new_changes']])

print(df.sample())
# df.to_parquet('new_changes_done_.parquet')