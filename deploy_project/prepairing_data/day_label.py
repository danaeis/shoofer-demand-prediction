import datetime
import pandas as pd


class DayLabelData():
    def __init__(self, dateset_path, end_date, start_date):
        self.END_DATE = datetime.date.fromisoformat(str(end_date))
        self.START_DATE = datetime.date.fromisoformat(str(start_date))
        self.INPUT_PATH = dateset_path
        
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
        labels_df = (
            location_date_df
            .merge(dataset_labels, how='left', on=['Location', 'Date'])
            .fillna(value=0)
        )
        
        # 
        prediction_dates = pd.DataFrame([self.END_DATE], columns=['Date'])

        prediction_rows_df = (
                            locations
                            .merge(prediction_dates, how='cross')
                            .sort_values(['Location', 'Date'])
                            .reset_index(drop=True)
                            )
        
        prediction_rows_df['Demand'] = 0

        labels_df = (pd.concat([labels_df,prediction_rows_df])
                          .sort_values(['Location', 'Date'], 
                                        ascending=[True, True])
                        )
        labels_df['Date'] = pd.to_datetime(labels_df['Date'])

        
        del locations
        del dates
        del location_date_df
        return labels_df

    def add_features(self, labels_df, features):
        self.data_features = labels_df.copy()
        self.data_features['Previous_day_demand'] = self.data_features.groupby(['Location'])['Demand'].shift(1)
        self.data_features['Previous_week_demand'] = self.data_features.groupby(['Location'])['Demand'].shift(7)
        self.data_features['Previous_2week_demand'] = self.data_features.groupby(['Location'])['Demand'].shift(14)
        self.data_features['Day_of_week'] = self.data_features['Date'].dt.dayofweek   
        self.data_features['Day_of_month'] = self.data_features['Date'].dt.day
        return self.data_features[list(labels_df.columns)+features], labels_df.columns
