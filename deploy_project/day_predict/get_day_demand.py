import pandas as pd
import datetime

class GetDayDemand():
    def __init__(self, prediction_path, prediction_location):
        self.prediction_path = prediction_path+str(prediction_location.date)+"_day_predicted.parquet"
        self.prediction_date = datetime.date.fromisoformat(str(prediction_location.date))
        self.locations = [int(l) for l in prediction_location.location_ids]

    def get_location_demand(self):
        
        demand = pd.read_parquet(self.prediction_path,
                                filters=[('Date','=',self.prediction_date),
                                    ('Location','in',self.locations)]).reset_index(drop=True)
        
        demand = demand.astype({'Predicted_demand':'int'})
        result_dict = {}
        for _, row in demand.iterrows():
            date = row['Date']
            location = row['Location']
            predicted_demand = row['Predicted_demand']

            if date not in result_dict:
                result_dict[date] = {}

            if location not in result_dict[date]:
                result_dict[date][location] = {}

            result_dict[date][location] = predicted_demand

        return result_dict

