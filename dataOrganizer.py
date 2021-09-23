import pandas as pd
import os


# Read in features
df = pd.read_parquet('FeaturesMonthlyEdu_2_US_Small_7eef3ffdfd-data.parquet')
# Read in the list of stock to be used
# Small
df_stock = pd.read_parquet('Funds_1_US_Small_9bc3c6ec30-data.parquet')
# Medium
# df_stock = pd.read_parquet('Funds_2_US_Small_4366060a0f-data.parquet')
# Large
# df_stock = pd.read_parquet('Funds_False_US_Small_1c8ad8da13-data.parquet')


# Directory
holding_directory = "holding"
noh_directory = "no_holding"

# Identify unique id and reaport dates in features data
funds = df['fund_id'].unique()
dates = df['report_date'].unique()

# All stock needed
for stock in df_stock["factset_fund_id"]:
    for fund in funds:
        if stock == fund:
            selected_1 = df.loc[df['fund_id'] == fund]

            for date in dates:
                selected_2 = selected_1.loc[selected_1['report_date'] == date]

                # Check if is_holding is all 0, return true
                #
                # if (selected_2["is_holding"] == 0).all():
                #     selected_2.to_parquet(os.path.join(noh_directory,'fund_{}_date_{}.parquet'.format(fund, str(date)[:10])))
                #
                # else:
                #     selected_2.to_parquet(os.path.join(holding_directory, 'fund_{}_date_{}.parquet'.format(fund, str(date)[:10])))

