import pandas as pd
from prepairing_data.labels import Label_Data
from training_models.xgb_model import Train_xgb_model

class Demand_Prediction_Model():
    def __init__(self, dataset_path, prediction_date, prediction_path):
        self.period = 3
        self.prediction_date = prediction_date
        self.predict_path = prediction_path
        labeling_data = Label_Data(dataset_path,self.prediction_date,self.period)
        self.labeled_dataset = labeling_data.labeling()
        self.model_dataset = labeling_data.add_features()

    def predict_model(self):
        self.train_dataset = self.model_dataset[~self.model_dataset['Demand'].isna()]
        self.predict_dataset = self.model_dataset[self.model_dataset['Demand'].isna()]
        training_model = Train_xgb_model(self.train_dataset, self.predict_dataset, self.predict_path)
        training_model.model_predict()
 
