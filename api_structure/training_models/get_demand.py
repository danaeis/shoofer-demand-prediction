import pandas as pd
import datetime

class Get_Demand():
    def __init__(self, prediction_path):
        self.prediction_path = prediction_path

    def get_location_demand(self, prediction_location):
        # prediction_interval = int(date.interval)
        prediction_date = datetime.date.fromisoformat(str(prediction_location.date))
        locations = [int(l) for l in prediction_location.location_ids]
        demand = pd.read_parquet(self.prediction_path,
                                filters=[('Date','=',prediction_date),
                                    ('Location','in',locations)]).reset_index(drop=True)
        
        demand = demand.astype({'Predicted_demand':'int'})
        return_value = demand[['Location','Predicted_demand']].to_dict('record')
        return return_value
