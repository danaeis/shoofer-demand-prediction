import pandas as pd
from prepairing_data.labels import Label_Data
from training_models.xgb_model import Train_xgb_model

class Demand_Prediction_Model():
    def __init__(self, prediction_date):
        self.period = 3
        self.prediction_date = prediction_date
        labeling_data = Label_Data(self.prediction_date,self.period)
        self.labeled_dataset = labeling_data.labeling()
        self.model_dataset = labeling_data.add_features()

    def check(self):
        print("in check", (self.labeled_dataset).shape)
        print('len nan',(self.model_dataset['Demand'].isna().sum()))
        return {
            'label loc shape':[len(self.labeled_dataset['Location'].unique())],
            'feature loc shape':[len(self.model_dataset['Location'].unique())],
            'feat date num':[len(self.model_dataset['Date'].unique())]
                
                }
        # print(self.model_dataset[self.model_dataset['Date']==pd.to_datetime(self.prediction_date)])
    def predict_model(self):
        self.train_dataset = self.model_dataset[~self.model_dataset['Demand'].isna()]
        self.predict_dataset = self.model_dataset[self.model_dataset['Demand'].isna()]
        training_model = Train_xgb_model(self.train_dataset, self.predict_dataset, self.prediction_date)
        training_model.model_predict()
        # return training_model
