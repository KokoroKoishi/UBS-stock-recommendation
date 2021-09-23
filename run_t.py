# Process time calculation
from time import process_time
# import plot package
import matplotlib.pyplot as plt
# Start the stopwatch / counter
t1_start = process_time()

# wandb
# Flexible integration for any Python script
import wandb
# wandb.login(key = "3d40ee7f3bd9794778bd2c5b0f175dd77827eb17")
# # 1. Start a W&B run
# wandb.init(entity='edu-ubs-nyu-2021q3', project='stock_recommendation', name ='kate_zhuang')

# 2. Save model inputs and hyperparameters
config = wandb.config
config.learning_rate = 0.01

# dataprep
import d6tflow, luigi
import pandas as pd
import numpy as np
import pathlib

# modeling
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.inspection import plot_partial_dependence
from sklearn.model_selection import cross_validate
import lightgbm
#lightgbm.LGBMRegressor
import shap
#shap.initjs()


# viz
import matplotlib.pyplot as plt
import seaborn as sns


# project
import cfg

import importlib # optional
importlib.reload(cfg)
# importlib.reload(tasks)

pd.set_option('display.expand_frame_repr', False)

import tasks
params = {}
params_model = cfg.params_model_default

params = {**params,**params_model}

#########################################
#########################################
#########################################
# EDA dataset
feature_importance = pd.read_excel('feature importance.xlsx')
shap = pd.read_excel('shap_value_eda.xlsx')
number_of_holdings = pd.read_excel("number_holdings.xlsx")
net_enter = pd.read_excel("enter_exit.xlsx")
f1 = pd.read_excel("f1.xls")
pred_prob = pd.read_excel("pred_prob_quantile.xlsx")
new_position = pd.read_excel("median of new holdings.xls")
exit_position = pd.read_excel("median of exit holdings.xls")
survival = pd.read_excel("survival.xlsx")
feature_box = pd.read_excel("features_summary.xlsx")

# Save EDA plot
filepath = '/Users/katez/Desktop/capstone/final_model'
folder = 'Picture/'
# EDA Plot functions
def feature_importance_plot(id):
  funds = feature_importance["Id"].unique()
  for fund in funds:
    if id == fund:
      selected = feature_importance.loc[feature_importance['Id'] == fund]
      selected.set_index(['Date'], inplace=True)
      selected.plot.area(stacked=True)
      plt.title(fund+" Feature importance")
      plt.xlabel("Date")
      plt.ylabel("Importance score")
      plt.show()
      filename = '{}_feature_importance.png'.format(fund)
      ch_filepath = filepath + '/' + folder + filename
      plt.savefig(ch_filepath)

def fund_shap_area(id):
    positions = ['hold', 'enter', 'exit']
    fund = shap.loc[shap['fund_id'] == id]
    for position in positions:
        plt.title(id + ' Shap value: ' + position)
        plt.stackplot(fund['date'], fund['oper_mgn_{}'.format(position)], fund['pe_{}'.format(position)],
                      fund['div_yld_{}'.format(position)], fund['roa_{}'.format(position)],
                      fund['roe_{}'.format(position)],
                      fund['pay_out_ratio_{}'.format(position)], fund['pbps_{}'.format(position)],
                      labels=['oper_mgn', 'pe', 'div_yld', 'roa', 'roe', 'pay_out_ratio', 'pbps'])
        plt.legend()
        plt.xlabel('date')
        plt.ylabel('{}_shap'.format(position))
        plt.show()
        filename = '{}_shap_value.png'.format(fund)
        ch_filepath = filepath + '/' + folder + filename
        plt.savefig(ch_filepath)

def num_of_holdings_plot(id):
  funds = number_of_holdings["fund_id"].unique()
  for fund in funds:
    if id == fund:
      selected = number_of_holdings.loc[number_of_holdings['fund_id'] == fund]
      selected.set_index(['report_date'], inplace=True)
      # selected.plot.area(stacked=True)
      selected.plot()
      plt.title(fund+" Number of holdings")
      plt.xlabel("Date")
      plt.ylabel("Number of holdings")
      plt.show()
      filename = '{}_number_of_holdings.png'.format(fund)
      ch_filepath = filepath + '/' + folder + filename
      plt.savefig(ch_filepath)

def num_of_net_enter_plot(id):
  funds = net_enter["fund_id"].unique()
  for fund in funds:
    if id == fund:
      data = net_enter.loc[net_enter["fund_id"] == fund]
      data.plot(x = "report_date", y = "number_enter/exit", kind = "line", xlabel = "Date", ylabel = "Net enter",title = fund+" Number of net enter")
      filename = '{}_number_of_enter.png'.format(fund)
      ch_filepath = filepath + '/' + folder + filename
      plt.savefig(ch_filepath)

def f1_plot(id):
  funds = f1['Fund'].unique()
  for fund in funds:
    if id == fund:
      data = f1.loc[f1['Fund']==fund]
      data.plot(x = "Date", y = ["F1"], kind="line", xlabel = "Date", ylabel = "F1 score", title = fund+" F1 score")
      filename = '{}_f1_Score.png'.format(fund)
      ch_filepath = filepath + '/' + folder + filename
      plt.savefig(ch_filepath)

def pred_prob_plot(id):
  funds = pred_prob['ID'].unique()
  for fund in funds:
    if id == fund:
      data = pred_prob.loc[pred_prob['ID']==fund].drop(columns = "ID")
      data = data.set_index("Time")
      plt.boxplot(data)
      plt.title(fund+" Predicted probability")
      plt.xlabel("Probability")
      plt.ylabel("Time")
      plt.show()
      filename = '{}_predict_probability.png'.format(fund)
      ch_filepath = filepath + '/' + folder + filename
      plt.savefig(ch_filepath)

def new_position_plot(id):
  funds_new = new_position['fund_id'].unique()
  for fund in funds_new:
    if id == fund:
      temp_plot = new_position.loc[new_position['fund_id']==fund]
      temp_plot = temp_plot[['report_date', 'roe', 'roa', 'oper_mgn', 'pay_out_ratio',
                            'pe', 'pbps', 'div_yld']]
      temp_plot.set_index(['report_date'], inplace=True)
      temp_plot.plot.area(stacked=True)
      plt.xlabel("Date")
      plt.ylabel("Features")
      plt.title(fund+" New position features")
      filename = '{}_new_position.png'.format(fund)
      ch_filepath = filepath + '/' + folder + filename
      plt.savefig(ch_filepath)

def exit_position_plot(id):
  funds_exit = exit_position['fund_id'].unique()
  for fund in funds_exit:
    if id == fund:
      temp_plot = exit_position.loc[exit_position['fund_id']==fund]
      temp_plot = temp_plot[['report_date', 'roe', 'roa', 'oper_mgn', 'pay_out_ratio',
                            'pe', 'pbps', 'div_yld']]
      temp_plot.set_index(['report_date'], inplace=True)
      temp_plot.plot.area(stacked=True)
      plt.xlabel("Date")
      plt.ylabel("Features")
      plt.title(fund+ " Exited position features")
      filename = '{}_exit_position.png'.format(fund)
      ch_filepath = filepath + '/' + folder + filename
      plt.savefig(ch_filepath)

def survival_plot(id):
  funds = survival["Fund"].unique()
  for fund in funds:
    if id == fund:
      selected = survival.loc[survival['Fund'] == fund]
      selected = selected.iloc[: , 1:]
      selected.set_index(['Date'], inplace=True)
      # selected['avg_target_probability'].plot()
      selected.plot()
      plt.title(fund+" Survival plot")
      plt.xlabel("Date")
      plt.ylabel("Average stock target probability")
      plt.show()
      filename = '{}_survival_plot.png'.format(fund)
      ch_filepath = filepath + '/' + folder + filename
      plt.savefig(ch_filepath)

def feature_boxplot(id):
  funds = feature_box['Fund'].unique()
  features = ['pbps','oper_mgn','pe','div_yld','roa','roe','pay_out_ratio']
  for fund in funds:
    selected_1 = feature_box.loc[feature_box['Fund']==fund]
    if (id == fund):
      for feature in features:
        transpose = pd.DataFrame()
        transpose_2 = pd.DataFrame()
        transpose_3 = pd.DataFrame()
        dates = selected_1['Date'].unique()
        # selected_2 = selected_1[['Fund','Date','{}_hold'.format(feature),'{}_enter'.format(feature),'{}_exit'.format(feature)]]
        selected_2 = selected_1[['Date','{}_hold'.format(feature)]]
        selected_enter = selected_1[['Date','{}_enter'.format(feature)]]
        selected_exit = selected_1[['Date','{}_exit'.format(feature)]]
        for date in dates:
          selected_3 = selected_2.loc[selected_2['Date']==date]
          selected_3 = selected_3.set_axis(['Date',date],axis=1)
          selected_3 = selected_3.drop(columns = 'Date')
          selected_3.reset_index(drop=True, inplace=True)
          transpose[date] = selected_3[date]

          selected_3_enter = selected_enter.loc[selected_enter['Date']==date]
          selected_3_enter = selected_3_enter.set_axis(['Date',date],axis=1)
          selected_3_enter = selected_3_enter.drop(columns = 'Date')
          selected_3_enter.reset_index(drop=True, inplace=True)
          transpose_2[date] = selected_3_enter[date]

          selected_3_exit = selected_exit.loc[selected_exit['Date']==date]
          selected_3_exit = selected_3_exit.set_axis(['Date',date],axis=1)
          selected_3_exit = selected_3_exit.drop(columns = 'Date')
          selected_3_exit.reset_index(drop=True, inplace=True)
          transpose_3[date] = selected_3_exit[date]

        transpose = transpose.T
        transpose_2 = transpose.T
        transpose_3 = transpose.T

        plt.title(fund+" hold feature: "+feature)
        plt.boxplot(transpose)
        plt.xlabel('Date')
        plt.ylabel('Feature value '+feature)
        plt.show()

        plt.title(fund+" enter feature: "+feature)
        plt.boxplot(transpose_2)
        plt.xlabel('Date')
        plt.ylabel('Feature value '+feature)
        plt.show()

        plt.title(fund+" exit feature: "+feature)
        plt.boxplot(transpose_3)
        plt.xlabel('Date')
        plt.ylabel('Feature value '+feature)
        plt.show()
        filename = '{}_feature_boxplot.png'.format(fund)
        ch_filepath = filepath + '/' + folder + filename
        plt.savefig(ch_filepath)
################################################################
################################################################

# Medium dataset
df_features = pd.read_parquet('new_changes_done.parquet')
# Identify unique id and report dates in features data
funds = df_features['fund_id'].unique()
dates = df_features['report_date'].unique()

# def recommend(fund, date):
# fund = funds[1]
# date = dates[2]
fund,dates = '04BG3Y-E','2020-05-31'
date = '{}T00:00:00.000000000'.format(dates)

# Find and assign the required data
selected = df_features[(df_features['fund_id'] == fund)&(df_features['report_date'] == date)]
selected = selected.set_index('stock_id')
# print(selected)
# selected['new_changes'].astype(int)
# print(selected["is_holding"].sum())
# print(selected["new_changes"].sum())
if selected["is_holding"].sum() <= 6 or selected["new_changes"].sum() <= 6:
  print("Not useable")
else:
  cfg.features_f = selected
  cfg.test = selected
  cfg.train = selected

  flow = d6tflow.Workflow(tasks.Screen,params)
  flow.preview(tasks.Screen)
  # TODO: Try to use the reset method to update code in tasks
  flow.reset(tasks.LoadTest,confirm=False)
  flow.reset(tasks.LoadTrain,confirm=False)
  flow.reset(tasks.Testinput,confirm=False)
  flow.reset(tasks.ModelInput,confirm=False)
  flow.reset(tasks.ModelTrain,confirm=False)
  flow.reset(tasks.Shap,confirm=False)
  flow.reset(tasks.ModelEval,confirm=False)
  flow.reset(tasks.Screen,confirm=False)

  flow.run(tasks.Screen)

  df_trainX, df_trainY = flow.outputLoad(tasks.ModelInput)

  # Modify data
  df_trainY = df_trainY.squeeze()
  cfg_col_X = df_trainX.columns
  cfg_col_Y = df_trainY.name

  mod_ols, mod_skols, mod_lgbm = flow.outputLoad(tasks.ModelTrain)
  explainer, shap_values, df_shap, dfo_shap = flow.outputLoad(tasks.Shap)
  df_train, df_importances = flow.outputLoad(tasks.ModelEval)
  df_screen, df_screen_enter, df_screen_exit = flow.outputLoad(tasks.Screen)

  print(df_screen_enter[0:5])
  print(df_screen_exit[0:5])
  print(tasks.f1(df_screen))

  ask = input("Do you want eda plot(yes/no): ")
  if ask == 'yes':
    feature_importance_plot(fund)
    fund_shap_area(fund)
    num_of_holdings_plot(fund)
    num_of_net_enter_plot(fund)
    f1_plot(fund)
    pred_prob_plot(fund)
    new_position_plot(fund)
    exit_position_plot(fund)
    survival_plot(fund)
    feature_boxplot(fund)




# recommend('04BG3Y-E','2020-05-31')




#######################
# fund = funds[1]
# date = dates[5]
# selected_2 = df_features.loc[(df_features['fund_id'] == fund)&(df_features['report_date'] == date)]
# selected_2 = selected_2.set_index('stock_id')