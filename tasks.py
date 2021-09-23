# dataprep
import d6tflow
import luigi
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import numpy as np
import warnings, keyring
import random

# modeling
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.inspection import plot_partial_dependence
from sklearn.model_selection import cross_validate
import lightgbm
#lightgbm.LGBMRegressor
import shap
#shap.initjs()
from imblearn.over_sampling import SMOTE

import cfg

class LoadTest(d6tflow.tasks.TaskPqPandas):
    def run(self):
        # Import file name from cfg
        # testwhole
        df = cfg.test
        self.save(df)

class LoadTrain(d6tflow.tasks.TaskPqPandas):
    def run(self):
        # Import file name from cfg
        #feature whole
        # df = pd.read_parquet(cfg.features_f)# read from export-Global Equity-11-24.parquet
        df = cfg.train
        self.save(df)


@d6tflow.requires(LoadTest)
class Testinput(d6tflow.tasks.TaskPqPandas):
    persist = ['xTest','yTest']

    def run(self):
        df_train = self.inputLoad()

        # Specify the columns for X and Y
        df_trainX = df_train[cfg.cfg_col_X]
        df_trainY = df_train[[cfg.cfg_col_Y]]

        self.save({'xTest':df_trainX, 'yTest':df_trainY})


@d6tflow.requires(LoadTrain)
class ModelInput(d6tflow.tasks.TaskPqPandas):
    persist = ['xTrain','yTrain']

    def run(self):
        df_train = self.inputLoad()

        # Specify the columns for X and Y
        df_trainX = df_train[cfg.cfg_col_X]
        df_trainY = df_train[[cfg.cfg_col_Y]]

        self.save({'xTrain':df_trainX, 'yTrain':df_trainY})

@d6tflow.requires(ModelInput)
class ModelTrain(d6tflow.tasks.TaskPickle):
    new_changes_smote = luigi.BoolParameter(default=False)
    new_changes_paste = luigi.BoolParameter(default=False)
    new_changes_ratio = luigi.FloatParameter(default = 0)


    imbalanced = luigi.BoolParameter(default=True)
    oversample = luigi.BoolParameter(default=True)
    lgbm_max_depth = luigi.IntParameter()
    lgbm_num_leaves = luigi.IntParameter()
    lgbm_learning_rate = luigi.FloatParameter()
    lgbm_n_estimators = luigi.IntParameter()
    lgbm_reg_alpha = luigi.FloatParameter()

    def run(self):
        df_trainX, df_trainY = self.inputLoad()
        df_trainY = df_trainY[cfg.cfg_col_Y]

        # Create smote datasets for balancing new_changes = 1 data
        if self.new_changes_smote:
            X_smote = df_trainX.drop(columns = ["new_changes"])
            X_smote['y'] = df_trainY
            y_smote = df_trainX["new_changes"]
            smote = SMOTE(random_state=1,sampling_strategy = self.new_changes_ratio)
            X_smote, y_smote = smote.fit_resample(X_smote, y_smote)

            if self.oversample:
                df_trainX = X_smote.drop(columns = 'y')
                df_trainY = X_smote['y']
                smote = SMOTE(random_state=1)
                df_trainX, df_trainY = smote.fit_resample(df_trainX, df_trainY)

        # Do not use smote
        if self.new_changes_paste:
            # Obtain a conbination of X and Y
            print("check")
            copy = df_trainX
            copy['is_holding'] = df_trainY

            new_changes = copy.loc[copy['new_changes'] == 1]

            a_1 = copy['new_changes'].value_counts()[1]
            a_2 = copy['new_changes'].value_counts()[0]
            # update by emma
            total_copy = int((self.new_changes_ratio*a_2))
            need_copy = total_copy-a_1

            # Calculate noise
            sem = new_changes.sem(axis=0)

            for i in range(need_copy):
                # make sure there is no NaN
                while True:
                    rand = random.uniform(-0.5,0.5)
                    if  rand != 0.0:
                        break
                temp = new_changes.sample()+(sem*rand)
                temp['is_holding'] = 1
                # copy = pd.concat(([copy, new_changes.sample()]))



            df_trainX = copy[["roe","roa","oper_mgn","pay_out_ratio","pe","pbps","div_yld"]]
            df_trainY = copy['is_holding']

            if self.oversample:
                smote = SMOTE(random_state=1)
                df_trainX, df_trainY = smote.fit_resample(df_trainX, df_trainY)

        if not self.new_changes_paste and not self.new_changes_smote:

            df_trainX = df_trainX.drop(columns=["new_changes"])
            if self.oversample:
                smote = SMOTE(random_state=1)
                df_trainX, df_trainY = smote.fit_resample(df_trainX, df_trainY)


        mod_ols = sm.Logit(df_trainY, sm.add_constant(df_trainX))

        params = {}
        if self.imbalanced:
            params['class_weight'] = "balanced"
        mod_skols = LogisticRegression(**params)
        mod_skols.fit(df_trainX,df_trainY)

        params = dict(max_depth=self.lgbm_max_depth,num_leaves=self.lgbm_num_leaves,learning_rate=self.lgbm_learning_rate,n_estimators=self.lgbm_n_estimators,reg_alpha=self.lgbm_reg_alpha,silent=True, random_state=0)
        if self.imbalanced:
            params['class_weight'] = "balanced"
        mod_lgbm = lightgbm.LGBMClassifier(**params)
        mod_lgbm.fit(df_trainX,df_trainY)

        self.save((mod_ols, mod_skols, mod_lgbm))




@d6tflow.requires(Testinput, ModelTrain)
class Shap(d6tflow.tasks.TaskPickle):

    def run(self):
        df_testX = self.input()[0]['xTest'].load()
        df_testY = self.input()[0]['yTest'].load()
        mod_ols, mod_skols, mod_lgbm = self.input()[1].load()

        df_testX = df_testX.drop(columns = "new_changes")

        explainer = shap.TreeExplainer(mod_lgbm, df_testX, model_output="probability")
        shap_values = explainer.shap_values(df_testX)
        df_shap = pd.DataFrame(shap_values, columns=df_testX.columns)
        dfo_shap = df_shap.clip(-0.1, 0.1)

        self.save((explainer, shap_values, df_shap, dfo_shap))

# change from here
@d6tflow.requires(ModelInput, ModelTrain, LoadTrain, LoadTest, Testinput)
class ModelEval(d6tflow.tasks.TaskPickle):

    def run(self):
        df_trainX = self.input()[0]['xTrain'].load()
        df_trainY = self.input()[0]['yTrain'].load()
        mod_ols, mod_skols, mod_lgbm = self.input()[1].load()
        df_train = self.input()[2].load()
        df_test=self.input()[3].load()
        df_testX= self.input()[4]['xTest'].load()
        df_testY = self.input()[4]['yTest'].load()

        df_trainX = df_trainX.drop(columns="new_changes")
        df_testX = df_testX.drop(columns = "new_changes")


        # feature importance
        df_importances = pd.DataFrame({'feature':df_testX.columns,'importance':np.round(mod_lgbm.feature_importances_,3)})
        # df_importances = df_importances.sort_values('importance',ascending=False).set_index('feature')

        # model predictions
        df_test['target_naive1'] = df_test[cfg.cfg_col_Y].value_counts().nlargest(n=1).index[0]  # most common class

        df_test['target_skols'] = mod_skols.predict(df_testX)
        df_test['target_skols_p'] = mod_skols.predict_proba(df_testX)[:, 1]


        df_test['target_lgbm']=mod_lgbm.predict(df_testX)
        df_test['target_lgbm_p']=mod_lgbm.predict_proba(df_testX)[:,1]
        df_test['target_predict']=df_test['target_lgbm']
        df_test['target_probability']=df_test['target_lgbm_p']


        self.save((df_test, df_importances))

@d6tflow.requires(ModelEval, Testinput, Shap, LoadTest)
class Screen(d6tflow.tasks.TaskPqPandas):
    persist = ['all','enter','exit']

    def run(self):
        df_test, df_importances = self.input()[0].load()
        df_testX = self.input()[1]['xTest'].load()
        df_testY = self.input()[1]['yTest'].load()
        df_testX = df_testX.drop(columns="new_changes")
        cfg_col_X = df_testX.columns
        explainer, shap_values, df_shap, dfo_shap = self.input()[2].load()
        df_testX_raw = self.input()[3].load()
        df_testX_raw = df_testX_raw[set(cfg_col_X).intersection(df_testX_raw)]



        df_screen = df_test[[cfg.cfg_col_Y,'target_lgbm','target_lgbm_p']]
        df_screen = df_screen.set_index(df_test.index)

        cfg_col_X_top = df_shap.std().sort_values(ascending=False).index.values[:20].tolist()
        dft_shap = df_shap[cfg_col_X_top].copy()
        dft_shap['model_probability_base'] = explainer.expected_value
        dft_shap['model_probability_chk'] = dft_shap['model_probability_base']+df_shap[cfg_col_X].sum(axis=1)
        dft_shap['others_impact'] = df_shap[cfg_col_X].sum(axis=1)-df_shap[cfg_col_X_top].sum(axis=1)
        dft_shap['model_probability'] = df_test['target_lgbm_p'].values
        # assert (dft_shap['model_probability_chk'].round(3)==dft_shap['model_probability'].round(3)).all()
        # assert (dft_shap['model_probability'].round(3)==dft_shap[['model_probability_base']+cfg_col_X_top+['others_impact']].sum(axis=1).round(3)).all()
        dft_shap = dft_shap.set_index(df_test.index)[['model_probability_base']+cfg_col_X_top+['others_impact','model_probability']]
        dft_shap = dft_shap.round(3)

        df_screen = df_screen.join(dft_shap)
        df_screen = df_screen.drop_duplicates()
        df_screen = df_screen.join(df_test.set_index(df_test.index)[cfg_col_X_top], lsuffix='_impact', rsuffix='_rank')

        df_screen = df_screen.join(df_testX_raw.set_index(df_test.index))
        df_screen = df_screen.rename(columns={'target_lgbm':'target_predict', 'target_lgbm_p':'target_probability'})
        df_screen = df_screen.sort_values('target_probability',ascending=False)

        df_screen_enter = df_screen[(df_screen[cfg.cfg_col_Y]==0)&(df_screen['target_predict']==1)].sort_values('target_probability',ascending=False)
        df_screen_exit = df_screen[df_screen[cfg.cfg_col_Y]==1].sort_values('target_probability')

        df_screen_enter = df_screen_enter.round(3)
        df_screen_exit = df_screen_exit.round(3)

        self.save({'all':df_screen,'enter':df_screen_enter,'exit':df_screen_exit})

# Calculate f1_score
def f1(df_train,weighted):
    y_true = df_train["is_holding"]
    y_pred = df_train["target_predict"]
    if weighted:
        return f1_score(y_true,y_pred,average = 'weighted')
    else:
        return f1_score(y_true, y_pred)

def each_holdings(df, funds, dates):
    number_each = []
    each_holdings_pd = pd.DataFrame()
    for fund in funds:
        for date in dates:
            stock = df.loc[df['report_date'] == date]
            stock = stock.loc[stock['fund_id'] == fund]
            number_each.append(stock['is_holding'].sum())
        each_holdings = pd.DataFrame(({'report_date': dates, 'number_holdings': number_each}))
        each_holdings.insert(0, "fund_id",fund, True)
        each_holdings_pd = each_holdings_pd.append(each_holdings)
        # each_holdings.plot(x='report_date', y= 'number_holdings',title = fund)
        # pyplot.show()
        number_each = []
        # print(each_holdings)
    return each_holdings_pd

#Code: each_holdings(df_fe_2).to_excel('number_holdings.xlsx', index = False)
#To create the excel file.

#Function to compute number of positions of Enter/Exit
def enter_exit_positions(df,funds,dates):
    number = []
    enter_exit_pd = pd.DataFrame()
    for fund in funds:
        for date in dates:
            stock = df.loc[df['report_date'] == date]
            stock = stock.loc[stock['fund_id'] == fund ]
            enter_stock = stock['is_holding'].shift(1) < stock['is_holding']
            count_enter = enter_stock.sum()
            number.append(count_enter)
        enter_exit_positions = pd.DataFrame(({'report_date': dates,'number_enter/exit': number}))
        enter_exit_positions.insert(0, "fund_id", fund, True)
        number = []
        enter_exit_positions['number_enter/exit'] = enter_exit_positions['number_enter/exit'].diff()
        enter_exit_positions = enter_exit_positions.dropna()
        enter_exit_pd = enter_exit_pd.append(enter_exit_positions)
        # enter_exit_positions.plot(x='report_date',y= 'number_enter/exit', title = fund)
        # pyplot.show()
        # print(enter_exit_positions)

    return enter_exit_pd

#Code: enter_exit_positions(df_fe_2).to_excel('enter_exit.xlsx', index = False)
#To create the excel file.
