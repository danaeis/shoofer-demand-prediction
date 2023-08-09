import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

class Train_ridge_regression_model():
    def __init__(self, train, predict, predict_path):
        self.PREDICT_PATH = predict_path
        self.train_dataset = train
        self.predict_dataset = predict
        self.AUTO_TUNE = False
        self.FEATURE_LIST = [
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
                model = Ridge()
                self.best_params, self.best_score = self.grid_search(
                    model, 
                    {'alpha':[661000,661200, 661400, 661600, 661800, 662000]},
                    self.train_dataset, 
                    cv = 5, 
                    feature_list = self.FEATURE_LIST
                    )

            else:
                self.best_params = {
                     'alpha': 661000
                     }
                

    def model_predict(self):
        self.tune_model()
        self.model = Ridge(**self.best_params)
        self.model.fit(self.train_dataset[self.FEATURE_LIST], self.train_dataset['Demand'])
        train_predict_demands  = self.model.predict(self.train_dataset[self.FEATURE_LIST])
        forecast_predict_demands  = self.model.predict(self.predict_dataset[self.FEATURE_LIST])

        train_predicted_df = self.train_dataset[['Date','Location']]
        train_predicted_df['Predicted_demand'] = train_predict_demands

        forecast_predicted_df = self.predict_dataset[['Date','Location']]
        forecast_predicted_df['Predicted_demand'] = forecast_predict_demands

        forecast_predicted_df['Date'] = forecast_predicted_df['Date'].astype('datetime64')
        train_predicted_df['Date'] = train_predicted_df['Date'].astype('datetime64')
        self.predicted_df = pd.concat([train_predicted_df,forecast_predicted_df])
        
        del forecast_predicted_df
        del train_predicted_df
        self.predicted_df.to_parquet(self.PREDICT_PATH+"ridge_predicted.parquet")
        