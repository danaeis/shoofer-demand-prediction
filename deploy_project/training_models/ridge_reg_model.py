import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge


class TrainRidgeRegressionModel():
    def __init__(self, train_df, predict_df, train_val_df, validation_df, features, prediction_path, auto_tune):
        self.train_df = train_df
        self.predict_df = predict_df
        self.train_val_df = train_val_df
        self.validation_df= validation_df
                
        self.PREDICT_PATH = prediction_path 

        self.AUTO_TUNE = auto_tune
        self.FEATURE_LIST = features

    def grid_search(self, model_class, param_grid, train_df, val_df):
        best_params = None
        best_val_loss = float('inf')

        for params in product(*param_grid.values()):
            current_params = dict(zip(param_grid.keys(), params))
            current_model = model_class(**current_params)
            current_model.fit(train_df[self.FEATURE_LIST], train_df['Demand'])
            
            y_val_pred = current_model.predict(val_df[self.FEATURE_LIST])
            val_loss = mean_squared_error(val_df['Demand'], y_val_pred)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = current_params
        
        return best_params, best_val_loss
    def tune_model(self):
            if self.AUTO_TUNE:
                model = Ridge
                self.best_params , self.best_score = self.grid_search(
                    model, 
                    {'alpha':[1, 10, 100],
                    },
                    self.train_val_df,
                    self.validation_df,
                    )

            else:
                self.best_params = {
                     'alpha': 661000
                     }
                

    def predict(self):
        self.tune_model()
        self.model = Ridge(**self.best_params)
        self.model.fit(self.train_df[self.FEATURE_LIST], self.train_df['Demand'])
        train_predict_demands  = self.model.predict(self.train_df[self.FEATURE_LIST])
        forecast_predict_demands  = self.model.predict(self.predict_df[self.FEATURE_LIST])

        self.train_predicted_df = self.train_df[['Date','Location']]
        self.train_predicted_df['Predicted_demand'] = train_predict_demands

        self.forecast_predicted_df = self.predict_df[['Date','Location']]
        self.forecast_predicted_df['Predicted_demand'] = forecast_predict_demands

    def save_as_feature(self):
        self.forecast_predicted_df['Date'] = pd.to_datetime(self.forecast_predicted_df['Date']).dt.date
        self.train_predicted_df['Date'] = pd.to_datetime(self.train_predicted_df['Date']).dt.date
        self.predicted_df = pd.concat([self.train_predicted_df,self.forecast_predicted_df])
        
        del self.forecast_predicted_df
        del self.train_predicted_df
        self.predicted_df.to_parquet(self.PREDICT_PATH+"_ridge.parquet")
        