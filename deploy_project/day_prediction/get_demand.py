import pandas as pd
import datetime

class GetDemand():
    def __init__(self, prediction_path, prediction_location):
        self.prediction_path = prediction_path+str(prediction_location.date)+"_predicted.parquet"
        self.prediction_date = datetime.date.fromisoformat(str(prediction_location.date))
        self.locations = [int(l) for l in prediction_location.location_ids]

    def get_location_demand(self):
        
        demand = pd.read_parquet(self.prediction_path,
                                filters=[('Date','=',self.prediction_date),
                                    ('Location','in',self.locations)]).reset_index(drop=True)
        
        demand = demand.astype({'Predicted_demand':'int'})
        return_value = demand[['Location','Predicted_demand']].to_dict('record')
        return {str(self.prediction_date):return_value}
