import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta  
from os.path import exists

from prepairing_data.day_label import LabelData
from training_models.ridge_reg_model import TrainRidgeRegressionModel
from training_models.xgb_model import TrainXgbModel


class DayDemandPredictionModel():
    def __init__(self, dataset_path, prediction_date, prediction_path):
        self.train_period = 3
        self.val_period = 7
        self.prediction_date = datetime.date.fromisoformat(str(prediction_date))
        self.start_train_date = self.prediction_date - relativedelta(months=self.train_period)
        self.start_validation_date = self.prediction_date - relativedelta(days=self.val_period)
        self.PREDICT_PATH = prediction_path + str(prediction_date)
        self.DATASET_PATH = dataset_path
        labeling_data = LabelData(self.DATASET_PATH, self.prediction_date, self.start_train_date)
        self.data_df, label_col = labeling_data.add_features()
        self.features = list(self.data_df.columns[len(label_col):])


    def split_dataset(self,dataset, split_date):
        split_date = datetime.date.fromisoformat(str(split_date))
        first_section = dataset[dataset['Date'] < split_date]
        second_section = dataset[dataset['Date'] >= split_date]

        return first_section, second_section

    def train_ridge_model(self):
        train_df, predict_df = self.split_dataset(self.data_df, self.prediction_date)
        train_df = train_df.dropna()
        train_val_df, validation_df = self.split_dataset(train_df, self.start_validation_date)
    
        training_model = TrainRidgeRegressionModel( train_df, 
                                                    predict_df, 
                                                    train_val_df, 
                                                    validation_df, 
                                                    self.features,
                                                    self.PREDICT_PATH, 
                                                    True)
        training_model.predict()
        training_model.save_as_feature()


    def add_predicted_to_feature(self):
        ridge_df = pd.read_parquet(self.PREDICT_PATH+"_ridge.parquet")
        self.data_df['Date']= pd.to_datetime(self.data_df['Date']).dt.date
        ridge_df['Date']=pd.to_datetime(ridge_df['Date']).dt.date
        data_ridge_df = (
                self.data_df
                .merge(ridge_df, how='left', on=['Location', 'Date'])
                .rename(columns = {'Predicted_demand' : 'Ridge_predict'})
                ).sort_values(['Location', 'Date'], ascending=[True, True]).reset_index(drop = True)
        self.features.append('Ridge_predict')

        self.train_df, self.predict_df = self.split_dataset(data_ridge_df, self.prediction_date)
        self.train_df = self.train_df.dropna()
        self.train_val_df, self.validation_df = self.split_dataset(self.train_df, self.start_validation_date)


    def train_xgb_model(self):
        self.train_ridge_model()
        self.add_predicted_to_feature()

        self.training_model = TrainXgbModel(self.train_df, 
                                       self.predict_df, 
                                       self.train_val_df, 
                                       self.validation_df,
                                       self.features, 
                                       self.PREDICT_PATH, 
                                       True)
    
    def predict_date(self):
        file_exists = exists(self.PREDICT_PATH+'_predicted.parquet')
        if file_exists:
            pass
        else:
            self.train_xgb_model()
            self.training_model.model_predict()