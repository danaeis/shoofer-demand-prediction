import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta  
from os.path import exists

from prepairing_data.day_label import DayLabelData
from training_models.ridge_reg_model import TrainRidgeRegressionModel
from training_models.xgb_model import TrainXgbModel
from .features import RF_FEATURE_LIST,RIDGE_FEATURE_LIST,XGB_FEATURE_LIST


class DayDemandPredictionModel():
    def __init__(self, dataset_path, prediction_date, prediction_path):
        self.train_period = 3
        self.val_period = 7
        self.prediction_date = datetime.date.fromisoformat(str(prediction_date))
        self.start_train_date = self.prediction_date - relativedelta(months=self.train_period)
        self.start_validation_date = self.prediction_date - relativedelta(days=self.val_period)
        self.PREDICT_PATH = prediction_path + str(prediction_date) + str('_day')
        self.DATASET_PATH = dataset_path
        
        self.labeling_obj = DayLabelData(self.DATASET_PATH, self.prediction_date, self.start_train_date)
        self.labels_df = self.labeling_obj.labeling()
        

    def split_dataset(self,dataset, split_date):
        dataset['Date'] = pd.to_datetime(dataset['Date']).dt.date
        split_date = datetime.date.fromisoformat(str(split_date))
        first_section = dataset[dataset['Date'] < split_date]
        second_section = dataset[dataset['Date'] >= split_date]

        return first_section, second_section

    def train_ridge_model(self):
        data_df, label_col = self.labeling_obj.add_features(self.labels_df, RIDGE_FEATURE_LIST)
        features = list(data_df.columns[len(label_col):])
        train_df, predict_df = self.split_dataset(data_df, self.prediction_date)
        train_df = train_df.dropna()
        train_val_df, validation_df = self.split_dataset(train_df, self.start_validation_date)
        params_test = {'alpha':[661000,661200, 661400, 661600, 661800, 662000]}
        best_params = {'alpha': 661000}
        training_model = TrainRidgeRegressionModel( train_df, 
                                                    predict_df, 
                                                    train_val_df, 
                                                    validation_df, 
                                                    features,
                                                    list(label_col)[:-1],
                                                    self.PREDICT_PATH, 
                                                    True,
                                                    params_test)
        training_model.predict()
        training_model.save_as_feature()


    def add_ridge_predicted_to_feature(self, data_df):
        ridge_df = pd.read_parquet(self.PREDICT_PATH+"_ridge.parquet")
        data_df['Date']= pd.to_datetime(data_df['Date']).dt.date
        ridge_df['Date']=pd.to_datetime(ridge_df['Date']).dt.date
        data_ridge_df = (
                data_df
                .merge(ridge_df, how='left', on=['Location', 'Date'])
                .rename(columns = {'Predicted_demand' : 'Ridge_predict'})
                ).sort_values(['Location', 'Date'], ascending=[True, True]).reset_index(drop = True)
        return data_ridge_df
        

    def train_xgb_model(self):
        data_df, label_col = self.labeling_obj.add_features(self.labels_df, RIDGE_FEATURE_LIST)
        features = list(data_df.columns[len(label_col):])
        self.train_ridge_model()
        print("after ridge trained")
        data_ridge_df = self.add_ridge_predicted_to_feature(data_df)
        features.append('Ridge_predict')
        
        train_df, predict_df = self.split_dataset(data_ridge_df, self.prediction_date)
        train_df = train_df.dropna()
        train_val_df, validation_df = self.split_dataset(train_df, self.start_validation_date)

        param_test = { 'learning_rate':[0.1], 
                        'subsample':[0.5, 0.6, 0.7, 0.8], 
                        'colsample_bytree':[0.7, 0.8, 0.9], 
                        'max_depth':[4, 5, 6, 7, 8], 
                        'min_child_weight':[5, 10, 20],
                    }
        
        best_params = {
                        'colsample_bytree': 0.8, 
                        'learning_rate': 0.1, 
                        'max_depth': 6, 
                        'min_child_weight': 20, 
                        'subsample': 0.9
                    }

        self.training_model = TrainXgbModel(train_df, 
                                            predict_df, 
                                            train_val_df, 
                                            validation_df,
                                            features,
                                            list(label_col)[:-1],
                                            self.PREDICT_PATH, 
                                            False,
                                            best_params)
    
    def predict_date(self):
        file_exists = exists(self.PREDICT_PATH+'_predicted.parquet')
        if file_exists:
            pass
        else:
            self.train_xgb_model()
            self.training_model.model_predict()