import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta  

class Label_Data():
    def __init__(self, end_date, period):
        self.END_DATE = datetime.date.fromisoformat(str(end_date))
        self.START_DATE = datetime.date.fromisoformat(str(self.END_DATE - relativedelta(months=period)))
        self.INPUT_PATH = '/home/saadi/DS/shoofer_demand_deploy/data/datasets/'
        
    def load_data(self):

        filtered_df = pd.read_parquet(self.INPUT_PATH,filters=[('tpep_pickup_datetime','<',self.END_DATE),
                                    ('tpep_pickup_datetime','>',self.START_DATE)])
        
        self.dataset = filtered_df.filter(items=['tpep_pickup_datetime', 'PULocationID'])
        self.dataset['PU_date'] = pd.to_datetime(self.dataset['tpep_pickup_datetime'].dt.date)

    def labeling(self):

        self.load_data()
        dataset_labels = (
            self.dataset
            .groupby(['PULocationID', 'PU_date'])['PU_date']
            .count()
            .to_frame('Demand')
            .sort_values(['PULocationID', 'PU_date'], ascending=[True, True])
            .reset_index()
            .rename(columns={'PULocationID': 'Location', 'PU_date': 'Date'})
        )
        locations = pd.DataFrame(dataset_labels['Location'].unique(), columns=['Location'])
        dates = pd.DataFrame(dataset_labels['Date'].unique(), columns=['Date'])
        location_date_df = (
            locations
            .merge(dates, how='cross')
            .sort_values(['Location', 'Date'], ascending=[True, True])
            .reset_index(drop=True)
        )
        self.labels_df = (
            location_date_df
            .merge(dataset_labels, how='left', on=['Location', 'Date'])
            .fillna(value=0)
        )
        predict_row_df = pd.DataFrame({'Location':locations['Location'],
                                       'Date':[self.END_DATE]*len(locations), 
                                       'Demand':[None]*len(locations)})
        self.labels_df = (pd.concat([self.labels_df,predict_row_df])
                          .sort_values(['Location', 'Date'], 
                                        ascending=[True, True])
                        )
        
        del locations
        del dates
        del location_date_df
        return self.labels_df

    def add_features(self):
        self.data_features = pd.DataFrame(columns=self.labels_df.columns)
        self.data_features = self.labels_df.copy()
        self.data_features['Previous_day_demand'] = self.data_features.groupby(['Location'])['Demand'].shift(1)
        self.data_features['Previous_week_demand'] = self.data_features.groupby(['Location'])['Demand'].shift(7)
        self.data_features['Previous_2week_demand'] = self.data_features.groupby(['Location'])['Demand'].shift(14)
        self.data_features['Day_of_week'] = pd.to_datetime(self.data_features['Date']).dt.dayofweek   
        self.data_features['Day_of_month'] = pd.to_datetime(self.data_features['Date']).dt.day
        self.data_features = self.data_features.sort_values(['Location', 'Date'], 
                                                            ascending=[True, True])

        return self.data_features
