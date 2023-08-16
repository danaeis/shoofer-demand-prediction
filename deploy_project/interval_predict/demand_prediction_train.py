import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta  
from os.path import exists

from prepairing_data.interval_label import IntervalLabelData
from training_models.ridge_reg_model import TrainRidgeRegressionModel
from training_models.xgb_model import TrainXgbModel
from training_models.rf_model import TrainRFModel 

from .features import XGB_FEATURE_LIST,RF_FEATURE_LIST,RIDGE_FEATURE_LIST


class IntervalDemandPredictionModel():
    def __init__(self, dataset_path, prediction_date, prediction_path):
        self.train_period = 3   #months
        self.val_period = 7   #days
        self.hour_interval = 3   #hours
        self.prediction_date = datetime.date.fromisoformat(str(prediction_date))
        self.start_train_date = self.prediction_date - relativedelta(months=self.train_period)
        self.start_validation_date = self.prediction_date - relativedelta(days=self.val_period)
        self.PREDICT_PATH = prediction_path + str(prediction_date) + str('_interval')
        self.DATASET_PATH = dataset_path

        self.labeling_obj = IntervalLabelData(self.DATASET_PATH, self.prediction_date, self.start_train_date, self.hour_interval)
        self.labels_df = self.labeling_obj.labeling()


    def split_dataset(self, dataset, split_date):
        dataset['Date'] = pd.to_datetime(dataset['Date']).dt.date
        split_date = datetime.date.fromisoformat(str(split_date))
        first_section = dataset[dataset['Date'] < split_date]
        second_section = dataset[dataset['Date'] >= split_date]

        return first_section, second_section

    def train_ridge_model(self):
        data_df, label_col = self.labeling_obj.add_features(self.labels_df, 14, RIDGE_FEATURE_LIST)
        features = list(data_df.columns[len(label_col):])
        train_df, predict_df = self.split_dataset(data_df, self.prediction_date)
        train_df = train_df.dropna()
        train_val_df, validation_df = self.split_dataset(train_df, self.start_validation_date)
        
        params_test = {'alpha':[100000,500000,1000000]}
        best_params = {'alpha': 500000}
        training_model = TrainRidgeRegressionModel( train_df, 
                                                    predict_df, 
                                                    train_val_df, 
                                                    validation_df, 
                                                    features,
                                                    list(label_col)[:-1],
                                                    self.PREDICT_PATH, 
                                                    False,
                                                    best_params)
        training_model.predict()
        training_model.save_as_feature()


    def add_ridge_predicted_to_feature(self, data_df):
        ridge_df = pd.read_parquet(self.PREDICT_PATH+"_ridge.parquet")
        data_df['Date']= pd.to_datetime(data_df['Date']).dt.date
        ridge_df['Date']=pd.to_datetime(ridge_df['Date']).dt.date
        data_ridge_df = (
                data_df
                .merge(ridge_df, how='left', on=['Location', 'Date','Hour_interval'])
                .rename(columns = {'Predicted_demand' : 'Ridge_predict'})
                ).sort_values(['Location', 'Date','Hour_interval'], ascending=[True, True, True]).reset_index(drop = True)
        return data_ridge_df
    
    def train_xgb_model(self):
        data_df, label_col = self.labeling_obj.add_features(self.labels_df, 14, XGB_FEATURE_LIST)
        features = list(data_df.columns[len(label_col):])

        self.train_ridge_model()
        data_ridge_df = self.add_ridge_predicted_to_feature(data_df)

        train_df, predict_df = self.split_dataset(data_ridge_df, self.prediction_date)
        train_df = train_df.dropna()
        train_val_df, validation_df = self.split_dataset(train_df, self.start_validation_date)

        param_test = { 'learning_rate':[0.05, 0.1, 0.15], 
                        'subsample':[0.7, 0.8, 0.9], 
                        'colsample_bytree':[0.6, 0.8, 0.9], 
                        'max_depth':[7, 8, 9, 10, 12], 
                        'min_child_weight':[10, 20, 5],
                        'n_estimators':[100]
                    }
        best_params = {'colsample_bytree': 0.9, 
                        'min_child_weight': 20, 
                        'learning_rate': 0.01, 
                        'max_depth': 10, 
                        'subsample': 0.8, 
                        'n_estimators': 500
                    }
        training_model = TrainXgbModel(train_df, 
                                            predict_df, 
                                            train_val_df, 
                                            validation_df,
                                            features,
                                            list(label_col)[:-1],
                                            self.PREDICT_PATH, 
                                            False,
                                            best_params)
        return training_model
        
    def train_rf_model(self):
        data_df, label_col = self.labeling_obj.add_features(self.labels_df, 14, RF_FEATURE_LIST)
        features = list(data_df.columns[len(label_col):])

        self.train_ridge_model()
        data_ridge_df = self.add_ridge_predicted_to_feature(data_df)
        features.append('Ridge_predict')

        train_df, predict_df = self.split_dataset(data_ridge_df, self.prediction_date)
        train_df = train_df.dropna()
        train_val_df, validation_df = self.split_dataset(train_df, self.start_validation_date)

        param_test ={
                        'oob_score': [True],
                        'max_features': [0.6, 0.7], 
                        'max_depth': [12, 14], 
                        'min_samples_leaf': [2, 5, 8],
                        'n_estimators': [300]
                    }
        best_params = {
                        'oob_score': True,
                        'max_features': 0.6, 
                        'max_depth': 14, 
                        'min_samples_leaf': 5,
                        'n_estimators': 500
                        }
        training_model = TrainRFModel(train_df, 
                                            predict_df, 
                                            train_val_df, 
                                            validation_df,
                                            features, 
                                            list(label_col)[:-1],
                                            self.PREDICT_PATH, 
                                            False,
                                            best_params)
        return training_model
    
    def predict_date(self):
        file_exists = exists(self.PREDICT_PATH+'_predicted.parquet')
        if file_exists:
            pass
        else:
            training_model = self.train_rf_model()
            training_model.model_predict()