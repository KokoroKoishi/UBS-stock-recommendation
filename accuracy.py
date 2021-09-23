# Process time calculation
from time import process_time
# Start the stopwatch / counter
t1_start = process_time()

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

#***************************************
# flow
#***************************************

import tasks
params = {}
params_model = cfg.params_model_default

f1_results=pd.DataFrame()
accuracy_results = pd.DataFrame()

# Read in features data
# Medium dataset
df_features = pd.read_parquet('new_changes_done_.parquet')
# Large dataset

# Identify unique id and reaport dates in features data
funds = df_features['fund_id'].unique()
dates = df_features['report_date'].unique()

# , 'new_changes' = True, 'new_changes_ratio' = 0.2
# put params_model in for loop bc theyll constantly be altered by for loop
# params_model = cfg.params_model_default
# params_model['new_changes_smote'] = True
# params_model['new_changes_smote_ratio'] = 0.1
# params = {**params, **params_model}

# update by jason
new_changes_ratio_array=[0.3,0.4,0.5,0.6,0.7]
for ratio in new_changes_ratio_array:
    accuracy_results = pd.DataFrame()

    for fund in funds:
        for i in range(0,len(dates)-1):
            params_model = cfg.params_model_default
            # update by jason
            params_model['new_changes_smote'] = False
            params_model['new_changes_paste'] = True
            params_model['new_changes_ratio'] = ratio

            selected = df_features.loc[(df_features['fund_id'] == fund)&(df_features['report_date'] == dates[i])]
            selected_2 = df_features.loc[(df_features['fund_id'] == fund)&(df_features['report_date'] == dates[i+1])]
            selected = selected.set_index('stock_id')
            selected_2 = selected_2.set_index('stock_id')
            if selected["is_holding"].sum() <= 6 :
                print("Not useable")
            else:
                cfg.features_f = selected
                cfg.test = selected
                cfg.train = selected

                # update by jason
                # adjust = 1/(len(selected)-selected["new_changes"].sum())
                # if selected["new_changes"].sum() <= 6 or selected["new_changes"].sum()>=(len(selected)-selected["new_changes"].sum())*(ratio-adjust):
                #
                #     params_model['new_changes_smote'] = False
                #     params_model['new_changes_paste'] = True
                #     print("unable to smote")
                if selected["new_changes"].sum() <1 or selected["new_changes"].sum()>=(len(selected)-selected["new_changes"].sum())*ratio:
                    params_model['new_changes_paste'] = False
                    print("unable to paste")
                params = {**params, **params_model}
                flow = d6tflow.Workflow(tasks.Screen, params)
                flow.preview(tasks.Screen)
                # TODO: Try to use the reset method to update code in tasks
                flow.reset(tasks.LoadTest, confirm=False)
                flow.reset(tasks.LoadTrain, confirm=False)
                flow.reset(tasks.Testinput, confirm=False)
                flow.reset(tasks.ModelInput, confirm=False)
                flow.reset(tasks.ModelTrain, confirm=False)
                flow.reset(tasks.Shap, confirm=False)
                flow.reset(tasks.ModelEval, confirm=False)
                flow.reset(tasks.Screen, confirm=False)

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
                print(df_train)

                # # f1
                # f1_results = f1_results.append({'Fund': fund, "Date": dates[i], 'F1': tasks.f1(df_train)},ignore_index=True)
                # accuracy
                df_acc = pd.DataFrame()
                df_acc['is_holding'] = df_train['is_holding']
                df_acc['target_predict'] = df_train['target_predict']

                next_month = selected_2[['is_holding']]
                next_month.rename(columns={'is_holding': 'next_holding'}, inplace=True)
                df_acc = pd.merge(df_acc, next_month, how='left', left_index=True, right_index=True)

                df_acc.loc[(df_acc['is_holding'] == 0) & (df_acc['target_predict'] == 1), 'pred_changes'] = 1
                df_acc.loc[df_acc['is_holding'] == 1, 'pred_changes'] = -1
                df_acc['pred_changes'] = df_acc['pred_changes'].fillna(0)

                df_acc.loc[(df_acc['is_holding'] == 0) & (df_acc['next_holding'] == 1), 'true_changes'] = 1
                df_acc.loc[df_acc['is_holding'] == 1, 'true_changes'] = -1
                df_acc['true_changes'] = df_acc['true_changes'].fillna(0)

                df_acc['accuracy'] = df_acc['pred_changes'] - df_acc['true_changes']

                accuracy_results = accuracy_results.append({'Fund': fund, 'Date': dates[i],'F1_weighted': tasks.f1(df_train,True), 'F1': tasks.f1(df_train, False),
                                                            'Accuracy': df_acc['accuracy'].value_counts()[0]/(len(df_acc['accuracy']))},ignore_index=True)
    accuracy_results.to_excel(str(ratio)+"paste_noise.xlsx")

