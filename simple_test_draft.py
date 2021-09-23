import numpy as np
import pandas as pd
import warnings, keyring

# modeling
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.inspection import plot_partial_dependence
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

from sklearn import metrics
import lightgbm
#lightgbm.LGBMRegressor
import shap
#shap.initjs()
from imblearn.over_sampling import SMOTE

df = pd.read_parquet('new_changes_done.parquet')

y_ = "is_holding"
X_ = ["roe","roa","oper_mgn","pay_out_ratio","pe","pbps","div_yld","new_changes"]

funds = df['fund_id'].unique()
dates = df['report_date'].unique()

fund = funds[1]
date = dates[4]
selected = df.loc[(df['fund_id'] == fund)&(df['report_date'] == date)]
selected = selected.set_index('stock_id')

fund = funds[1]
date = dates[5]
selected_2 = df.loc[(df['fund_id'] == fund)&(df['report_date'] == date)]
selected_2 = selected_2.set_index('stock_id')

print(selected.shape)
# selected = df

X = selected[X_]
y = selected[[y_]]

X_smote = X.drop(columns = "new_changes")
X_smote['y'] = y
y_smote = X["new_changes"]

smote = SMOTE(random_state=1,sampling_strategy = 0.1)
X_smote, y_smote = smote.fit_resample(X_smote, y_smote)

smote = SMOTE(random_state=1)
X_train = X_smote.drop(columns = 'y')
y_train = X_smote['y']
X_train, y_train = smote.fit_resample(X_train, y_train)

print(X.shape)
print(X_smote.shape)
print(y_smote.shape)
print(X_train.shape)
print(y_train.shape)

# print(y_train.value_counts())
# print(X_train)


mod_ols = sm.Logit(y_train, sm.add_constant(X_train))

mod_skols = LogisticRegression(class_weight="balanced")
mod_skols.fit(X_train, y_train)

mod_lgbm = lightgbm.LGBMClassifier(max_depth=5,num_leaves=10,learning_rate=0.1,
              n_estimators=50,reg_alpha=10.0,silent=True, random_state=0,class_weight='balanced')
# mod_lgbm = lightgbm.LGBMClassifier(max_depth=3,num_leaves=31,learning_rate=0.1,
#               n_estimators=50,reg_alpha=10.0,silent=True, random_state=0,class_weight='balanced')
mod_lgbm.fit(X_train, y_train)

# from sklearn.ensemble import GradientBoostingClassifier
# mod_lgbm = GradientBoostingClassifier(max_depth=3,learning_rate=0.1,
#                n_estimators=50,random_state=0)
# mod_lgbm.fit(X_train, y_train)

y_pred = mod_lgbm.predict(X.drop(columns = "new_changes"))
y_prob = mod_lgbm.predict_proba(X.drop(columns = "new_changes"))[:,1]
test = X
test['y'] = y
test['y_pred'] = y_pred
test['y_prob'] = y_prob

print('f1', f1_score(y,y_pred,average = 'weighted'))

# print(type(y_prob))
print(metrics.classification_report(y, y_pred))
# enter
# test_enter = test[(test['y'] == 0) & (test['y_pred'] == 1)].sort_values('y_prob', ascending=False)
# test_exit = test[test['y'] == 1].sort_values('y_prob', ascending=False)



# print(y_test.value_counts())


# print(test_enter)
# print(test_exit)
# print((478/(478+10))*0.91+(10/(478+10))*0.09)
# metrics.classification_report(y_test, y_pred)
# lightgbm.Dataset()

test.loc[(test['y'] == 0) & (test['y_pred'] == 1), 'pred_changes'] = 1
test.loc[(test['y'] == 1), 'pred_changes'] = -1
test['pred_changes'] = test['pred_changes'].fillna(0)

test['y_next'] = selected_2['is_holding']

test.loc[(test['y'] == 0) & (test['y_next'] == 1), 'true_changes'] = 1
test.loc[(test['y'] == 1), 'true_changes'] = -1
test['true_changes'] = test['true_changes'].fillna(0)

test['accuracy'] = test['pred_changes'] - test['true_changes']
test['pred_changes'] = test['pred_changes'].astype(int)
test['true_changes'] = test['true_changes'].astype(int)



# print(test['accuracy'].value_counts())
# print(test['pred_changes'].value_counts())
# print(test['true_changes'].value_counts())
print('action accuracy: ', test['accuracy'].value_counts()[0]/(test['accuracy'].value_counts()[0]+test['accuracy'].value_counts()[1]))
# print('f1',f1_score(test['pred_changes'],test['true_changes'],average = 'weighted'))
# metrics.classification_report(test['pred_changes'],test['true_changes'])
# print(selected.index)
# print(selected_2.index)
# assert(selected.index == selected_2.index)