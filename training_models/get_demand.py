import pandas as pd
import datetime

class Get_Demand():
    def __init__(self):
        self.PREDICTION_PATH = '/home/saadi/DS/shoofer_demand_deploy/data/predictions/predicted.parquet'

    def get_location_demand(self, date, location_id):
        prediction_interval = int(date.interval)
        prediction_date = datetime.date.fromisoformat(str(date.date))
        demand = pd.read_parquet(self.PREDICTION_PATH,
                                 filters=[('Date','=',prediction_date),
                                        ('Location','=',int(location_id))]).reset_index(drop=True)
        
        return {"location":location_id,"predicted_demand":int(demand['Predicted_demand'])}
