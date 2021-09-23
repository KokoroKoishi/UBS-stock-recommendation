# Process time calculation
from time import process_time
# Start the stopwatch / counter
t1_start = process_time()

# wandb
# Flexible integration for any Python script
# import wandb
# wandb.login(key = "3d40ee7f3bd9794778bd2c5b0f175dd77827eb17")
# # 1. Start a W&B run
# wandb.init(entity='edu-ubs-nyu-2021q3', project='stock_recommendation', name ='kate_zhuang')

# 2. Save model inputs and hyperparameters
# config = wandb.config
# config.learning_rate = 0.01

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
params_model = cfg.params_model_default # {'lgbm_max_depth':3,'lgbm_num_leaves':31,'lgbm_learning_rate':0.1,'lgbm_n_estimators':50,'lgbm_reg_alpha':10.0}
# for model in cfg.params_models:
# Official project
# run = wandb.init(reinit = True, project='stock_recommendation', entity='edu-ubs-nyu-2021q3')

# Local project
# run = wandb.init(reinit=True, project='stock-recommendation', entity='zhiheng-wang')
# run.name = model

# params_model = cfg.params_models[model]
params = {**params,**params_model}

# Results dataframe
f1_results=[]

# features_importance_result = pd.DataFrame()
# each_holdings_result = pd.DataFrame()
# enter_exit_positions_result = pd.DataFrame()


# funds_enter_featurebox = {}
# funds_exit_featurebox = {}
# funds_featurebox = {}
# summary_featurebox_result = pd.DataFrame()

new_holdings = pd.DataFrame()
exited_holdings = pd.DataFrame()

# output_new = pd.DataFrame()
# output_exit = pd.DataFrame()

# survival_result = pd.DataFrame()
#
# probability_result = pd.DataFrame()
#
# shap_hold_result = []
# shap_enter_result = []
# shap_exit_result = []

# Read in features data
# Medium dataset
df_features = pd.read_parquet('FeaturesMonthlyEdu_2_US_Small_7eef3ffdfd-data.parquet')
# Large dataset

# Identify unique id and reaport dates in features data
funds = df_features['fund_id'].unique()
dates = df_features['report_date'].unique()



# newly_portion settings for mannnually adjustments
for newly_portion in [0,0.05,0.1,0.2]:
    f1_result = pd.DataFrame()
    for fund in funds:
        # fund = funds[0]
        selected_1 = df_features.loc[df_features['fund_id'] == fund]
        # fund_enter_featurebox = pd.DataFrame()
        # fund_exit_featurebox = pd.DataFrame()
        # fund_featurebox = pd.DataFrame()
        fund_enter_pred_prob = pd.DataFrame()
        temp = pd.DataFrame()
        # temp for last month data
        for date in dates:
            #date = dates[0]
            selected_2 = selected_1.loc[selected_1['report_date'] == date]
            # summary_withoutdate_featurebox = pd.DataFrame()
            if selected_2["is_holding"].sum() <= 6:
                print("Not useable")
            else:


                if (temp.shape[0] > 0):
                    # find out if this is the first month, if it is not:
                    new = pd.DataFrame()
                    # for new stocks in general
                    temp["full_id"] = temp[['fsym_id', 'fsym_regional_id']].agg('_'.join, axis=1)
                    selected_2["full_id"] = selected_2[['fsym_id', 'fsym_regional_id']].agg('_'.join, axis=1)
                    new = selected_2[~selected_2["full_id"].isin(temp["full_id"])]
                    withoutnew = selected_2[~selected_2["full_id"].isin(new["full_id"].values)]
                    # withoutnew=selected_2[selected_2["full_id"].str.contains(new["full_id"].values)==False]
                    for i in range(len(withoutnew)):
                    # to find out newly entered, and newly exited position
                        stock = withoutnew["full_id"].iloc[i]
                        array = temp.loc[temp["full_id"] == stock]["is_holding"].values
                        flag = int(withoutnew["is_holding"].iloc[i]) - array[0]

                        if (flag != 0):
                            new = pd.concat([new, withoutnew.iloc[[i]]])
                    new.drop(["full_id"], axis=1, inplace=True)
                    selected_2.drop(["full_id"], axis=1, inplace=True)
                    temp = selected_2
                    if len(new)!=0:

                        times = int(len(selected_2) * newly_portion // len(new))
                        # concat enough times to fill the newly_potion
                        for i in range(times):
                            selected_2 = pd.concat([selected_2, new])
                else:
                    # first month case
                    temp=selected_2






                cfg.features_f = selected_2

                flow = d6tflow.Workflow(tasks.Screen,params)
                flow.preview(tasks.Screen)
                # TODO: Try to use the reset method to update code in tasks
                flow.reset(tasks.FeaturesNormalized,confirm=False)
                flow.reset(tasks.ModelInput,confirm=False)
                flow.reset(tasks.ModelTrain,confirm=False)
                flow.reset(tasks.Shap,confirm=False)
                flow.reset(tasks.ModelEval,confirm=False)
                flow.reset(tasks.Screen,confirm=False)

                flow.run(tasks.Screen)

                df_train = flow.outputLoad(tasks.FeaturesNormalized)
                df_trainX, df_trainY = flow.outputLoad(tasks.ModelInput)
                # Modify data
                df_trainY = df_trainY.squeeze()
                cfg_col_X = df_trainX.columns
                cfg_col_Y = df_trainY.name

                mod_ols, mod_skols, mod_lgbm = flow.outputLoad(tasks.ModelTrain)
                explainer, shap_values, df_shap, dfo_shap = flow.outputLoad(tasks.Shap)
                df_train, df_importances = flow.outputLoad(tasks.ModelEval)
                df_screen, df_screen_enter, df_screen_exit = flow.outputLoad(tasks.Screen)



                # # Results saving
                # 1. F1 score
                # print("This is df_train\n", df_train)
                # print("f1_score for dataset {}_{} is".format(fund, str(date)[:10]), tasks.f1(df_train))
                f1_result = f1_result.append({"Portion": newly_portion, 'Fund': fund, "Date":date, 'F1': tasks.f1(df_train)},
                                             ignore_index=True)
    f1_result.to_csv(str(newly_portion)+".csv")
    f1_results.append(f1_result)

# 'Date': str(date)[:10],
# 2. Feature importance
# print(df_importances)
# importance = lightgbm.plot_importance(mod_lgbm)
# plt.show()
# column_name_fi = '{}_{}'.format(fund, str(date)[:10])
# features_importance_result[column_name_fi] = df_importances["importance"]
#
# # 3. Predicted probability
# df_features_X = df_trainX.loc[df_screen_enter.index]
# fund_enter_pred_prob[date] = [df_screen_enter['target_probability'].min(),
#                               df_screen_enter['target_probability'].quantile(.25),
#                               df_screen_enter['target_probability'].quantile(.5),
#                               df_screen_enter['target_probability'].quantile(.75),
#                               df_screen_enter['target_probability'].max()]
#
# # 4. Shap value
# # index in to get the SHAP values for the prediction of "1"(is_holding=true)
# # print(shap_values[1])
# # f = plt.figure(num = column_name)
# # shap.summary_plot(dfo_shap, df_trainX, plot_type="bar",show=False)
# # f.savefig("Instant{}_{}.png".format(fund, str(date)[:10]), bbox_inches='tight', dpi=600)
#
# holdings_shap = df_screen.loc[df_screen['is_holding']==1].iloc[:5,4:11]
# # print(holdings_shap)
# enter_shap = df_screen_enter.iloc[:5,4:11]
# # print(enter_shap)
# exit_shap = df_screen_exit.iloc[:5,4:11]
# # print(exit_shap)
#
# # index in to get the SHAP values for the prediction of "1"(is_holding=true)
# holdings_shap = df_screen.loc[df_screen['is_holding'] == 1].iloc[:5, 4:11]
# # print(holdings_shap)
# holdings_shap.loc['{}_{}'.format(fund, str(date)[:10])] = holdings_shap.mean()
# shap_hold_result.append(holdings_shap)
# # print(shap_result)
#
# enter_shap = df_screen_enter.iloc[:5, 4:11]
# # print(enter_shap)
# enter_shap.loc['{}_{}'.format(fund, str(date)[:10])] = enter_shap.mean()
# shap_enter_result.append(enter_shap)
#
# exit_shap = df_screen_exit.iloc[:5, 4:11]
# exit_shap.loc['{}_{}'.format(fund, str(date)[:10])] = exit_shap.mean()
# shap_exit_result.append(exit_shap)
# # print(exit_shap)
#
# # 5. number of holdings (below)
# # 6. number of positions enter,exit(below)
#
# # 7. feature boxplot
# date = pd.to_datetime(date)
# print(date.strftime('%m/%d/%Y'))
# # get the features of enter hold and exit
# dfenter_features_X = df_trainX.loc[df_screen_enter.index]
# df_features_X = df_trainX.loc[df_screen.index]
# dfexit_features_X = df_trainX.loc[df_screen_exit.index]
# for string in cfg.cfg_col_X:
#     summary_withoutdate_featurebox[string + '_hold'] = [df_features_X[string].min(),
#                                                      df_features_X[string].quantile(.25),
#                                                      df_features_X[string].quantile(.5),
#                                                      df_features_X[string].quantile(.75),
#                                                      df_features_X[string].max()]
# for string in cfg.cfg_col_X:
#     summary_withoutdate_featurebox[string + '_exit'] = [dfexit_features_X[string].min(),
#                                                      dfexit_features_X[string].quantile(.25),
#                                                      dfexit_features_X[string].quantile(.5),
#                                                      dfexit_features_X[string].quantile(.75),
#                                                      dfexit_features_X[string].max()]
# for string in cfg.cfg_col_X:
#     summary_withoutdate_featurebox[string + '_enter'] = [dfenter_features_X[string].min(),
#                                                       dfenter_features_X[string].quantile(.25),
#                                                       dfenter_features_X[string].quantile(.5),
#                                                       dfenter_features_X[string].quantile(.75),
#                                                       dfenter_features_X[string].max()]
# summary_withoutdate_featurebox.set_axis([fund + date.strftime('%m/%d/%Y')] * 5, axis=0, inplace=True)
# summary_featurebox_result = pd.concat([summary_featurebox_result, summary_withoutdate_featurebox])
#
# # 8. new positions median area chart (below)
# # 9. exit positions median area chart (below)
# # 10. survival plot
# # print(df_screen.columns)
# column_name_sp = '{}_{}'.format(fund, str(date)[:10])
# survival_result = survival_result.append({'Fund': fund, 'Date': str(date)[:10], 'avg_target_probability': df_screen["target_probability"].mean()},
#                              ignore_index=True)
# mean = df_screen["target_probability"].mean()
# survival_result[column_name_sp] = mean
# fund_enter_pred_prob=fund_enter_pred_prob.T
# fund_enter_pred_prob.insert(0, "stock", fund)
# probability_result = probability_result.append(fund_enter_pred_prob)
#
# shap_hold_result = pd.concat(shap_hold_result)
# shap_enter_result = pd.concat(shap_enter_result)
# shap_exit_result = pd.concat(shap_exit_result)

# print(f1_result)
# f1_result.to_csv('f1_result_adjusted_with_10%.csv')
# 5. number of holdings
# each_holdings_result = tasks.each_holdings(df_features, funds, dates)
#
# # 6. number of positions enter,exit
# enter_exit_positions_result = tasks.enter_exit_positions(df_features,funds,dates)
#
# # 8. new positions median area chart
# # 9. exit positions median area chart
# for fund in funds:
#     selected_1 = df_features.loc[df_features['fund_id'] == fund]
#     for i in range(1, len(dates)):
#         selected_2 = selected_1.loc[(selected_1['report_date'] == dates[i - 1]) & (selected_1['is_holding'] == 1)]
#         temp = selected_1.loc[(selected_1['report_date'] == dates[i]) & (selected_1['is_holding'] == 1)]
# #
#         # new holdings
#             old = selected_2['fsym_id'].unique()
#             new = temp['fsym_id'].unique()
#             enter = []
#             exit = []
#             for x in new:
#                 if (x not in old):
#                     enter.append(x)
#             new_adj = [x for x in new if x not in enter]
#             for y in new_adj:
#                 temp = temp.drop(index=(temp.loc[(temp['fsym_id'] == y)].index))
#             new_holdings = new_holdings.append(temp)
#     #         # exited holdings
#             for m in old:
#                 if (m not in new):
#                     exit.append(m)
#             old_adj = [x for x in old if x not in exit]
#             for n in old_adj:
#                 selected_2 = selected_2.drop(index=(selected_2.loc[(selected_2['fsym_id'] == n)].index))
#             exited_holdings = exited_holdings.append(selected_2)
#
# funds_new = new_holdings['fund_id'].unique()
# funds_exit = exited_holdings['fund_id'].unique()
#
# for fund in funds_new:
#     temp_new = new_holdings[new_holdings['fund_id']==fund]
#     output_temp = pd.DataFrame(columns=('fund_id', 'report_date', 'roe', 'roa',
#                                     'oper_mgn', 'pay_out_ratio', 'pe', 'pbps',
#                                     'div_yld'))
#     for i in range(len(dates)):
#         temp = temp_new[temp_new['report_date']==dates[i]]
#         row = {'fund_id':fund, 'report_date':dates[i], 'roe':temp['roe'].median(), 'roa':temp['roa'].median(),
#                 'oper_mgn':temp['oper_mgn'].median(), 'pay_out_ratio':temp['pay_out_ratio'].median(),
#                 'pe':temp['pe'].median(), 'pbps':temp['pbps'].median(), 'div_yld':temp['div_yld'].median()}
#         output_temp.loc[i] = row
#     output_new = output_new.append(output_temp)
#
# for fund in funds_exit:
#     temp_exit = exited_holdings[exited_holdings['fund_id']==fund]
#     output_temp_1 = pd.DataFrame(columns=('fund_id', 'report_date', 'roe', 'roa',
#                                       'oper_mgn', 'pay_out_ratio', 'pe', 'pbps',
#                                       'div_yld'))
#     for i in range(len(dates)):
#         temp_1 = temp_exit[temp_exit['report_date']==dates[i]]
#         row_1 = {'fund_id':fund, 'report_date':dates[i], 'roe':temp_1['roe'].median(), 'roa':temp_1['roa'].median(),
#                 'oper_mgn':temp_1['oper_mgn'].median(), 'pay_out_ratio':temp_1['pay_out_ratio'].median(),
#                 'pe':temp_1['pe'].median(), 'pbps':temp_1['pbps'].median(), 'div_yld':temp_1['div_yld'].median()}
#         output_temp_1.loc[i] = row_1
#     output_exit = output_exit.append(output_temp_1)
#
#
# features_importance_result.set_index(df_importances["feature"], inplace=True)
#
# # output to the screen
# print("1. f1 score", f1_result)
# print("2. feature importance",features_importance_result)
# print("3. predicted probability",probability_result)
# print("4. shap",shap_hold_result,shap_enter_result,shap_exit_result)
# print("5. number of holdings",each_holdings_result)
# print("6. number of positions enter, exit",enter_exit_positions_result)
# print("7. feature boxplot",summary_featurebox_result)
# print("8. new positions",output_new)
# print("9. exit positions",output_exit)
# print("10. survival plot",survival_result)
#
#
# # output to excel
# # Create a new excel workbook
# writer = pd.ExcelWriter('results.xlsx', engine='xlsxwriter')
#
# # 1. f1 score", f1_result)
# f1_result.to_excel(writer, sheet_name='f1 score')
# # 2. feature importance
# features_importance_result.to_excel(writer, sheet_name='feature importance')
# # 3. predicted probability
# probability_result.to_excel(writer, sheet_name='predicted probability')
# # 4. shap
# shap_hold_result.to_excel(writer, sheet_name='shap_hold_result.xls')
# shap_enter_result.to_excel(writer, sheet_name='shap_enter_result.xls')
# shap_exit_result.to_excel(writer, sheet_name='shap_exit_result.xls')
# # 5. number of holdings",each_holdings_result)
# each_holdings_result.to_excel(writer, sheet_name='number of holdings')
# # 6. number of positions enter, exit
# enter_exit_positions_result.to_excel(writer, sheet_name='number of net enter')
# # 7. feature boxplot
# summary_featurebox_result.to_excel(writer, sheet_name='feature boxplot')
# # 8. new positions
# output_new.to_excel(writer, sheet_name='new position median')
# # 9. exit positions
# output_exit.to_excel(writer, sheet_name='exit position median')
# # 10. survival plot
# survival_result.to_excel(writer, sheet_name='survival plot')
#
# # Close the Pandas Excel writer and output the Excel file.
# writer.save()
#
#
#
#
# # wandb push and finish
# # features_importance_result.to_excel("feature_importance.xls")
# # run.log({'df_trainX': wandb.Table(dataframe=df_trainX)})
# # run.log({'df_trainY': wandb.Table(dataframe=df_trainY)})
#
# # wandb.log({'top5holding_shap{}_{}'.format(fund, str(date)[:10]): wandb.Table(dataframe=holdings_shap)})
# # wandb.log({'top5enter_shap{}_{}'.format(fund, str(date)[:10]): wandb.Table(dataframe=enter_shap)})
# # wandb.log({'top5exit_shap{}_{}'.format(fund, str(date)[:10]): wandb.Table(dataframe=exit_shap)})
#
# # wandb.log({'feature_importance': wandb.Table(dataframe=features_importance_result)})
# # run.finish()