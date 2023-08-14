import xgboost as xgb
from itertools import product
from sklearn.metrics import mean_squared_error


class TrainXgbModel():
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
                

                params_test = {'learning_rate':[0.1], 
                                'subsample':[0.5, 0.6, 0.7, 0.8], 
                                'colsample_bytree':[0.7, 0.8, 0.9], 
                                'max_depth':[4, 5, 6, 7, 8], 
                                'min_child_weight':[5, 10, 20],
                            }

                self.best_params, self.best_score = self.grid_search(
                                                        model_class = xgb.XGBRegressor, 
                                                        param_grid = params_test,
                                                        train_df = self.train_val_df, 
                                                        val_df = self.validation_df
                                                        )
                
            else:
                self.best_params = {
                                    'colsample_bytree': 0.8, 
                                    'learning_rate': 0.1, 
                                    'max_depth': 6, 
                                    'min_child_weight': 20, 
                                    'subsample': 0.9
                                    }
                

    def model_predict(self):
        self.tune_model()
        self.model = xgb.XGBRegressor(**self.best_params)
        self.model.fit(self.train_df[self.FEATURE_LIST], self.train_df['Demand'])
        predict_demands  = self.model.predict(self.predict_df[self.FEATURE_LIST])
        self.predicted_df = self.predict_df[['Date','Location']]
        self.predicted_df['Predicted_demand'] = predict_demands
        self.predicted_df.to_parquet(self.PREDICT_PATH+"_predicted.parquet")
        