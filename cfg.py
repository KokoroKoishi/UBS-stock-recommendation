import pandas as pd
# File names
features_f = pd.DataFrame()# Test run
# features_f = "Funds_1_US_Small_9bc3c6ec30-data.parquet"

cfg_col_Y = "is_holding"
cfg_col_X = ["roe","roa","oper_mgn","pay_out_ratio","pe","pbps","div_yld","new_changes"]

test = pd.DataFrame()
train = pd.DataFrame()
# Default models
# params_model_default = {'lgbm_max_depth': 3, 'lgbm_num_leaves': 31, 'lgbm_learning_rate': 0.1, 'lgbm_n_estimators': 50,
#                         'lgbm_reg_alpha': 10.0}
params_model_default = {'lgbm_max_depth': 5, 'lgbm_num_leaves': 10, 'lgbm_learning_rate': 0.1, 'lgbm_n_estimators': 50,
                        'lgbm_reg_alpha': 10.0}



# Models description
params_models = {
    "baseline": {'lgbm_max_depth': 3, 'lgbm_num_leaves': 31, 'lgbm_learning_rate': 0.1, 'lgbm_n_estimators': 50,
                 'lgbm_reg_alpha': 10.0}
    # "depth = 4": {'lgbm_max_depth': 4, 'lgbm_num_leaves': 31, 'lgbm_learning_rate': 0.1, 'lgbm_n_estimators': 50,
    #               'lgbm_reg_alpha': 10.0},
    # "depth = 5": {'lgbm_max_depth': 5, 'lgbm_num_leaves': 31, 'lgbm_learning_rate': 0.1, 'lgbm_n_estimators': 50,
    #               'lgbm_reg_alpha': 10.0},
    # "leaves = 51": {'lgbm_max_depth': 3, 'lgbm_num_leaves': 51, 'lgbm_learning_rate': 0.1, 'lgbm_n_estimators': 50,
    #                 'lgbm_reg_alpha': 10.0},
    # "leaves = 81": {'lgbm_max_depth': 3, 'lgbm_num_leaves': 81, 'lgbm_learning_rate': 0.1, 'lgbm_n_estimators': 50,
    #                 'lgbm_reg_alpha': 10.0},
    # "rate = 0.2": {'lgbm_max_depth': 3, 'lgbm_num_leaves': 31, 'lgbm_learning_rate': 0.2, 'lgbm_n_estimators': 50,
    #                'lgbm_reg_alpha': 10.0},
    # "rate = 0.3": {'lgbm_max_depth': 3, 'lgbm_num_leaves': 31, 'lgbm_learning_rate': 0.3, 'lgbm_n_estimators': 50,
    #                'lgbm_reg_alpha': 10.0},
    # "estmr = 100": {'lgbm_max_depth': 3, 'lgbm_num_leaves': 31, 'lgbm_learning_rate': 0.1, 'lgbm_n_estimators': 100,
    #                 'lgbm_reg_alpha': 10.0},
    # "estmr = 150": {'lgbm_max_depth': 3, 'lgbm_num_leaves': 31, 'lgbm_learning_rate': 0.1, 'lgbm_n_estimators': 150,
    #                 'lgbm_reg_alpha': 10.0},
    # "alpha = 15": {'lgbm_max_depth': 3, 'lgbm_num_leaves': 31, 'lgbm_learning_rate': 0.1, 'lgbm_n_estimators': 50,
    #                'lgbm_reg_alpha': 15.0},
    # "alpha = 20": {'lgbm_max_depth': 3, 'lgbm_num_leaves': 31, 'lgbm_learning_rate': 0.1, 'lgbm_n_estimators': 50,
    #                'lgbm_reg_alpha': 20.0}
}
