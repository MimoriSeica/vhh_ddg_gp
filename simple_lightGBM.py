import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

def predict(data_x, data_y, test_x):
    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.20, random_state=0)
    lgbm_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1
    }
    lgb_train = lgb.Dataset(train_x, train_y, params={'verbose': -1})
    lgb_eval = lgb.Dataset(valid_x, valid_y, params={'verbose': -1})
    evals_result = {}
    gbm = lgb.train(params=lgbm_params,
                    train_set=lgb_train, 
                    valid_sets=[lgb_eval], 
                    early_stopping_rounds=10, 
                    evals_result=evals_result, 
                    verbose_eval=False);

    return gbm.predict(test_x), np.array(gbm.feature_importance())