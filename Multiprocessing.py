from multiprocessing import Process
from multiprocessing import Pool
from threading import Thread
import os,time
# dataprep
import d6tflow, luigi
import pandas as pd
import numpy as np
import pathlib
import numba




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
df_features = pd.read_parquet('new_changes_done.parquet')
# Large dataset

# Identify unique id and reaport dates in features data
funds = df_features['fund_id'].unique()
dates = df_features['report_date'].unique()

def run(funds,dates,params,accuracy_results):
    for fund in funds:
        for i in range(0, len(dates) - 1):
            params_model = cfg.params_model_default
            params_model['new_changes_paste'] = True
            params_model['new_changes_paste_ratio'] = 0.1
            # params_model['new_changes_paste_ratio']=new_changes_paste_ratio
            params = {**params, **params_model}
            selected = df_features.loc[(df_features['fund_id'] == fund) & (df_features['report_date'] == dates[i])]
            selected_2 = df_features.loc[
                (df_features['fund_id'] == fund) & (df_features['report_date'] == dates[i + 1])]
            selected = selected.set_index('stock_id')
            selected_2 = selected_2.set_index('stock_id')
            if selected["is_holding"].sum() <= 6:
                print("Not useable")
            else:
                cfg.features_f = selected
                cfg.test = selected
                cfg.train = selected

                if selected["new_changes"].sum() <= 6 or len(selected["new_changes"].value_counts()) < 2:
                    params_model['new_changes_smote'] = False
                    params_model['new_changes_smote_ratio'] = 0
                    params = {**params, **params_model}
                    print("unable to smote")

                if selected["new_changes"].sum() < 1 or selected["new_changes"].sum() > len(selected) / 4:
                    params_model['new_changes_paste'] = False
                    params_model['new_changes_paste_ratio'] = 0
                    params = {**params, **params_model}
                    print("unable to paste")

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

                # # f1
                # f1_results = f1_results.append({'Fund': fund, "Date": dates[i], 'F1': tasks.f1(df_train)},ignore_index=True)
                # accuracy
                df_acc = pd.DataFrame()
                df_acc['is_holding'] = df_train['is_holding']
                df_acc['target_predict'] = df_train['target_predict']
                df_acc['next_holding'] = selected_2['is_holding']

                df_acc.loc[(df_acc['is_holding'] == 0) & (df_acc['target_predict'] == 1), 'pred_changes'] = 1
                df_acc.loc[df_acc['is_holding'] == 1, 'pred_changes'] = -1
                df_acc['pred_changes'] = df_acc['pred_changes'].fillna(0)

                df_acc.loc[(df_acc['is_holding'] == 0) & (df_acc['next_holding'] == 1), 'true_changes'] = 1
                df_acc.loc[df_acc['is_holding'] == 1, 'true_changes'] = -1
                df_acc['true_changes'] = df_acc['true_changes'].fillna(0)

                df_acc['accuracy'] = df_acc['pred_changes'] - df_acc['true_changes']

                accuracy_results = accuracy_results.append(
                    {'Fund': fund, 'Date': dates[i], 'F1_weighted': tasks.f1(df_train, True),
                     'F1': tasks.f1(df_train, False),
                     'Accuracy': df_acc['accuracy'].value_counts()[0] / (len(df_acc['accuracy']))}, ignore_index=True)

    print(accuracy_results)
    accuracy_results.to_excel("accuracy.xlsx")

# def work():
#     res=0
#     for i in range(100000000):
#         res *=i



if __name__ == '__main__':
    l=[]
    cpu = os.cpu_count()
    print(cpu) #Print number of CPU threads of current computer
    start=time.time()
    for i in range(8):
        p=Process(target=run, args= (funds,dates,params,accuracy_results))
        # p = Thread(target=work)
        l.append(p)
        p.start()
    for p in l:
        p.join()
    stop=time.time()
    print('run time is %s' %(stop-start))



# if __name__=='__main__':
#     start = time.time()
#     p = Pool(4)
#     for i in range(5):
#         p.apply_async(work, args=())
#     print('等待所有子进程完成。')
#     p.close()
#     p.join()
#     end = time.time()
#     print("总共用时{}秒".format((end - start)))


