import pandas as pd
from prepairing_data.labels import Label_data
# from xgb_model import Train_xgb_model

class Demand_Prediction_Model():
    def __init__(self, prediction_date):
        self.period = 1
        self.prediction_date = prediction_date
        labeling_data = Label_data(self.prediction_date,self.period)
        self.labeled_dataset = labeling_data.labeling()
        # print(self.labeled_dataset)
        self.model_dataset = labeling_data.add_features()
        # self.model_dataset = None

    def check(self):
        print("in check", (self.labeled_dataset).shape)
        print('len nan',(self.model_dataset['Demand'].isna().sum()))
        return {
            'label loc shape':[len(self.labeled_dataset['Location'].unique())],
            'feature loc shape':[len(self.model_dataset['Location'].unique())],
            'feat date num':[len(self.model_dataset['Date'].unique())]
                
                }
        # print(self.model_dataset[self.model_dataset['Date']==pd.to_datetime(self.prediction_date)])
    # def predict_model(self):
    #     training_model = Train_xgb_model(self, self.model_dataset, self.prediction_date)
