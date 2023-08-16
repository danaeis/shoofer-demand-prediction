import xgboost as xgb
from itertools import product
from sklearn.metrics import mean_squared_error


class TrainXgbModel():
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
                self.best_params, self.best_score = self.grid_search(
                                                        model_class = xgb.XGBRegressor, 
                                                        param_grid = self.model_params,
                                                        train_df = self.train_val_df, 
                                                        val_df = self.validation_df
                                                        )
                
            else:
                self.best_params = self.model_params
                

    def model_predict(self):
        self.tune_model()
        self.model = xgb.XGBRegressor(**self.best_params)
        self.model.fit(self.train_df[self.FEATURE_LIST], self.train_df['Demand'])
        predict_demands  = self.model.predict(self.predict_df[self.FEATURE_LIST])
        self.predicted_df = self.predict_df[self.LABELS_COL]
        self.predicted_df['Predicted_demand'] = predict_demands
        self.predicted_df.to_parquet(self.PREDICT_PATH+"_predicted.parquet")
        