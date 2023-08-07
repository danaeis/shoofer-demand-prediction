import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

class Train_xgb_model():
    def __init__(self, train, predict, prediction_date):
        self.prediction_date = prediction_date
        self.PREDICT_PATH = '/home/saadi/DS/shoofer_demand_deploy/data/predictions/predicted.parquet'
        self.train_dataset = train
        self.predict_dataset = predict
        self.AUTO_TUNE = False
        self.FEATURE_LIST = [
                # 'ARIMA_predicts', 
                'Previous_2week_demand',
                'Previous_week_demand', 
                'Previous_day_demand', 
                'Day_of_month', 
                'Day_of_week'
                ]

    def grid_search(self, model, test_parameters, train_data, feature_list, cv = None):
        gs = GridSearchCV(
            estimator = model, 
            param_grid = test_parameters, 
            scoring = 'neg_root_mean_squared_error', 
            cv = cv, 
            n_jobs = -1
            )
        
        gs.fit(train_data[feature_list], train_data['Demand'])
        return gs.best_params_, gs.best_score_

    def tune_model(self):
            if self.AUTO_TUNE:
                params_test = {'learning_rate':[0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01], 
                            'subsample':[0.8, 0.9, 1], 
                            'colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1], 
                            'max_depth':[7], 
                            'gamma':[0, 1, 5],
                            'n_estimators':[1000]
                            }
                params = {"objective": "reg:squarederror"}

                self.best_params, self.best_score = self.grid_search(
                                                        model = xgb.XGBRegressor(**params), 
                                                        test_parameters = params_test,
                                                        train_data = self.train_df, 
                                                        feature_list = self.FEATURE_LIST, 
                                                        cv = 3
                                                        )
                
            else:
                self.best_params = {'colsample_bytree': 0.6, 
                            'gamma': 10, 
                            'learning_rate': 0.01, 
                            'max_depth': 7, 
                            'subsample': 0.9, 
                            'n_estimators': 1000
                            }
                

    def model_predict(self):
        self.tune_model()
        self.model = xgb.XGBRegressor(**self.best_params)
        self.model.fit(self.train_dataset[self.FEATURE_LIST], self.train_dataset['Demand'])
        predict_demands  = self.model.predict(self.predict_dataset[self.FEATURE_LIST])
        self.predicted_df = self.predict_dataset[['Date','Location']]
        self.predicted_df['Predicted_demand'] = predict_demands
        self.predicted_df.to_parquet(self.PREDICT_PATH)
        