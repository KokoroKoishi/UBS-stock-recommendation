# Process time calculation
from time import process_time
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

# Medium dataset
df_features = pd.read_parquet('new_changes_done.parquet')
# Identify unique id and reaport dates in features data
funds = df_features['fund_id'].unique()
dates = df_features['report_date'].unique()

fund = funds[1]
date = dates[4]
# prompt from user

# Find and assign the required data
selected = df_features[(df_features['fund_id'] == fund)&(df_features['report_date'] == date)]
selected = selected.set_index('stock_id')
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

# print(df_screen.columns)

print(tasks.f1(df_screen,True))
print("recommend: ", df_screen_enter.head())

#######################
# fund = funds[1]
# date = dates[5]
# selected_2 = df_features.loc[(df_features['fund_id'] == fund)&(df_features['report_date'] == date)]
# selected_2 = selected_2.set_index('stock_id')