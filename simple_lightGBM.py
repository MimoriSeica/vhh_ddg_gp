import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

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

def predict_2(data_x, data_y, test_x):
    n_splits = 4
    pred_y = None
    importance = None
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1234)
    
    for index, (train_indices, valid_indices) in enumerate(kf.split(range(data_x.shape[0]))):
        train_x, valid_x = data_x[train_indices], data_x[valid_indices] 
        train_y, valid_y = data_y[train_indices], data_y[valid_indices] 
    
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
        
        if pred_y is None:
            pred_y = gbm.predict(test_x)
            importance = np.array(gbm.feature_importance())
        else:
            pred_y += gbm.predict(test_x)
            importance += np.array(gbm.feature_importance())

    return pred_y / n_splits, importance