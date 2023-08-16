import datetime
import pandas as pd

class IntervalLabelData():
    def __init__(self, dateset_path, end_date, start_date, hour_interval):
        self.END_DATE = datetime.date.fromisoformat(str(end_date))
        self.START_DATE = datetime.date.fromisoformat(str(start_date))
        self.INPUT_PATH = dateset_path
        self.interval = hour_interval
        
    def load_data(self):

        filtered_df = pd.read_parquet(self.INPUT_PATH,filters=[('tpep_pickup_datetime','<',self.END_DATE),
                                    ('tpep_pickup_datetime','>',self.START_DATE)])
        
        self.dataset = filtered_df.filter(items=['tpep_pickup_datetime', 'PULocationID'])
        self.dataset['PU_date'] = pd.to_datetime(self.dataset['tpep_pickup_datetime'].dt.date)


    def labeling(self):
        self.load_data()
        dataset_labels = (
            self.dataset
            .groupby([self.dataset['PU_date'].dt.date,(self.dataset['tpep_pickup_datetime'].dt.hour//self.interval)*self.interval,'PULocationID'])['PULocationID']
            .count()
            .to_frame('Demand')
            .sort_values(['PULocationID','PU_date','tpep_pickup_datetime'])
            .reset_index()
            .rename(columns={'PULocationID': 'Location', 'PU_date': 'Date', 'tpep_pickup_datetime': 'Hour_interval'})
        )
        
        
        locations = pd.DataFrame(dataset_labels['Location'].unique(), columns=['Location'])
        dates = pd.DataFrame(dataset_labels['Date'].unique(), columns=['Date'])
        hour = pd.DataFrame(dataset_labels['Hour_interval'].unique(), columns=['Hour_interval'])\
            .sort_values('Hour_interval').reset_index(drop=True)
        
        
        location_date_df = (
            locations
            .merge(dates, how='cross')
            .sort_values(['Location', 'Date'])
            .reset_index(drop=True)
        )
        
        
        location_date_hour_df = (
            location_date_df
            .merge(hour, how='cross')
            .sort_values(['Location', 'Date', 'Hour_interval'])
            .reset_index(drop=True)
        )
        
        labels_df = (
            location_date_hour_df
            .merge(dataset_labels, how='left', on=['Location', 'Date', 'Hour_interval'])
            .fillna(value=0)
        )

        prediction_dates = pd.DataFrame([self.END_DATE], columns=['Date'])

        prediction_rows_df = (
                            locations
                            .merge(prediction_dates, how='cross')
                            .sort_values(['Location', 'Date'])
                            .reset_index(drop=True)
                            )
        prediction_rows_df = (
                            prediction_rows_df
                            .merge(hour, how='cross')
                            .sort_values(['Location','Date', 'Hour_interval'])
                            .reset_index(drop=True)
                            )
        prediction_rows_df['Demand'] = 0

        labels_df = (pd.concat([labels_df,prediction_rows_df])
                          .sort_values(['Location', 'Date', 'Hour_interval'], 
                                        ascending=[True, True, True])
                        )
        labels_df['Date'] = pd.to_datetime(labels_df['Date'])

        del locations
        del dates
        del location_date_df
        return labels_df
    
    def add_features(self, labels_df, lag_num, features):
        self.data_features = labels_df.copy()
        intervals_per_day = 24//self.interval
        self.data_features['previous_day_interval'] = self.data_features.groupby(['Location'])['Demand'].shift(1*intervals_per_day)
        self.data_features['previous_week_interval'] = self.data_features.groupby(['Location'])['Demand'].shift(7*intervals_per_day)
        self.data_features['previous_2week_interval'] = self.data_features.groupby(['Location'])['Demand'].shift(14*intervals_per_day)

        for i in range(1,lag_num):
            if i not in(1,7):
                self.data_features[f'previous_{i}day_interval'] = self.data_features.groupby('Location')['Demand'].shift(i*intervals_per_day)

        
        self.data_features['day_of_week'] = self.data_features['Date'].dt.dayofweek
        self.data_features['day_of_month'] = self.data_features['Date'].dt.day
        self.data_features['time'] = self.data_features['Hour_interval']
        self.data_features['zone'] = self.data_features['Location']

        df = self.data_features.sort_values(['Location','Date','Hour_interval'])[['Location','Date','Hour_interval','Demand']]        
        df['max_previous_week_interval'] = self.data_features.groupby(['Location'])['Demand'].shift(1).rolling(window = 7*intervals_per_day).max().values
        self.data_features['max_previous_week_interval'] = df.sort_values(['Location', 'Date','Hour_interval'])['max_previous_week_interval']
        df['max_previous_2week_interval'] = self.data_features.groupby(['Location'])['Demand'].shift(1).rolling(window = 14*intervals_per_day).max().values
        self.data_features['max_previous_2week_interval'] = df.sort_values(['Location', 'Date','Hour_interval'])['max_previous_2week_interval']

        
        df = self.data_features.sort_values(['Location','Date','Hour_interval'])[['Location','Date','Hour_interval','Demand']]
        df['max_previous_7exact_interval'] = self.data_features.groupby(['Location','Hour_interval'])['Demand'].shift(1).rolling(window = 7).max().values
        self.data_features['max_previous_7exact_interval'] = df.sort_values(['Location', 'Date','Hour_interval'])['max_previous_7exact_interval']
        df['max_previous_14exact_interval'] = self.data_features.groupby(['Location','Hour_interval'])['Demand'].shift(1).rolling(window = 14).max().values
        self.data_features['max_previous_14exact_interval'] = df.sort_values(['Location', 'Date','Hour_interval'])['max_previous_14exact_interval']

        
        self.data_features['previous_day_9interval'] = self.data_features.groupby('Location')['Demand'].shift(intervals_per_day+1)
        self.data_features['previous_day_10interval'] = self.data_features.groupby('Location')['Demand'].shift(intervals_per_day+2)
        self.data_features['previous_day_11interval'] = self.data_features.groupby('Location')['Demand'].shift(intervals_per_day+3)
        self.data_features['previous_day_12interval'] = self.data_features.groupby('Location')['Demand'].shift(intervals_per_day+4)
        self.data_features['previous_day_13interval'] = self.data_features.groupby('Location')['Demand'].shift(intervals_per_day+5)
        self.data_features['previous_day_14interval'] = self.data_features.groupby('Location')['Demand'].shift(intervals_per_day+6)
        self.data_features['previous_day_15interval'] = self.data_features.groupby('Location')['Demand'].shift(intervals_per_day+7)
        self.data_features['previous_day_17interval'] = self.data_features.groupby('Location')['Demand'].shift(intervals_per_day+9)

        
        self.data_features['avrg_previous_2day_8interval'] = (8*self.data_features['previous_day_9interval']+7*self.data_features['previous_day_10interval']+\
        6*self.data_features['previous_day_11interval']+5*self.data_features['previous_day_12interval']+4*self.data_features['previous_day_13interval']+\
        3*self.data_features['previous_day_14interval']+2*self.data_features['previous_day_15interval']+self.data_features['previous_2week_interval'])/intervals_per_day

        
        self.data_features['diff_previous_2day_previous_interval'] = self.data_features['previous_day_17interval']-self.data_features['previous_day_9interval']

        
        self.data_features['diff_previous_2day_interval'] = self.data_features['previous_day_interval']-self.data_features['previous_2day_interval']
        self.data_features['diff_previous_2week_interval'] = self.data_features['previous_week_interval']-self.data_features['previous_2week_interval']

        return self.data_features[list(labels_df.columns)+features], labels_df.columns
