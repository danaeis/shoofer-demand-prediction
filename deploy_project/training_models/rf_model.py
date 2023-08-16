from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class TrainRFModel():
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
        best_score = float('inf')
        best_params = None
        param_combination = product(*param_grid.values())
        
        for param in param_combination:

            param_dict = dict(zip(param_grid.keys(), param))
            model = model_class(**param_dict)
            model.fit(train_df[self.FEATURE_LIST], train_df['Demand'])
            
            validation_predict_df = model.predict(val_df[self.FEATURE_LIST])
            score = mean_squared_error(val_df['Demand'], validation_predict_df)
            
            if score<best_score:
                best_score = score
                best_params = param_dict
                
        return best_params, best_score


    def tune_model(self):
            if self.AUTO_TUNE:
                self.best_params, self.best_score = self.grid_search(
                                                                    model_class = RandomForestRegressor,
                                                                    test_parameters = self.model_params,
                                                                    train_data = self.train_val_df,
                                                                    validation_data = self.validation_df,
                                                                    feature_list = self.FEATURE_LIST
                                                                    )
                
            else:
                self.best_params = self.model_params
                

    def model_predict(self):
        self.tune_model()
        self.model = RandomForestRegressor(**self.best_params)
        self.model.fit(self.train_df[self.FEATURE_LIST], self.train_df['Demand'])
        predict_demands  = self.model.predict(self.predict_df[self.FEATURE_LIST])
        self.predicted_df = self.predict_df[self.LABELS_COL]
        self.predicted_df['Predicted_demand'] = predict_demands
        self.predicted_df.to_parquet(self.PREDICT_PATH+"_predicted.parquet")
        