import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge


class TrainRidgeRegressionModel():
    def __init__(self, train_df, predict_df, train_val_df, validation_df, features, label_col, prediction_path, auto_tune, param_test):
        self.train_df = train_df
        self.predict_df = predict_df
        self.train_val_df = train_val_df
        self.validation_df= validation_df
        self.FEATURE_LIST = features
        self.LABELS_COL = label_col
        self.AUTO_TUNE = auto_tune
        self.model_params = param_test
        self.PREDICT_PATH = prediction_path 


    def grid_search(self, model_class, param_grid, train_df, val_df):
        print("in gridsearch in train ridge regression model")
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
            print("in tune model in train ridge")
            if self.AUTO_TUNE:
                model = Ridge
                self.best_params , self.best_score = self.grid_search(
                    model, 
                    self.model_params,
                    self.train_val_df,
                    self.validation_df,
                    )

            else:
                self.best_params = self.model_params
                

    def predict(self):
        print("in ridge model predict")
        self.tune_model()
        print("after tune ridge")
        print("features",self.FEATURE_LIST)
        self.model = Ridge(**self.best_params)
        self.model.fit(self.train_df[self.FEATURE_LIST], self.train_df['Demand'])
        print(self.train_df[self.FEATURE_LIST].shape)
        print(self.train_df['Demand'].shape)
        print(self.predict_df[self.FEATURE_LIST].shape)
        print(self.model)
        self.train_predict_demands  = self.model.predict(self.train_df[self.FEATURE_LIST])
        self.forecast_predict_demands  = self.model.predict(self.predict_df[self.FEATURE_LIST])


    def save_as_feature(self):
        self.train_predicted_df = self.train_df[self.LABELS_COL]
        self.train_predicted_df['Predicted_demand'] = self.train_predict_demands

        self.forecast_predicted_df = self.predict_df[self.LABELS_COL]
        self.forecast_predicted_df['Predicted_demand'] = self.forecast_predict_demands

        self.forecast_predicted_df['Date'] = pd.to_datetime(self.forecast_predicted_df['Date']).dt.date
        self.train_predicted_df['Date'] = pd.to_datetime(self.train_predicted_df['Date']).dt.date
        self.predicted_df = pd.concat([self.train_predicted_df,self.forecast_predicted_df])
        
        del self.forecast_predicted_df
        del self.train_predicted_df
        self.predicted_df.to_parquet(self.PREDICT_PATH+"_ridge.parquet")
        