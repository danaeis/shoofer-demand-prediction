
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import warnings

from day_prediction.demand_prediction_train import DayDemandPredictionModel
from day_prediction.get_demand import GetDemand

warnings.filterwarnings('ignore')


class prediction_date(BaseModel):
    date: str

class prediction_date_location(BaseModel):
    date: str
    location_ids: list[int]

# pydantic handle date input
# pydantic handle location in range
app = FastAPI()

PREDICTION_PATH = '/home/saadi/DS/shoofer_demand_deploy/deploy_project/data/predictions/'
DATASET_PATH = '/home/saadi/DS/shoofer_demand_deploy/deploy_project/data/datasets'

@app.get("/")
def root():                                                                                               
    return {"API Provided": {
        "train model for day prediction":"/train_demand_predict",
        "get predicted demand":"/get_demand"
    }}

@app.post("/train_day_demand_predict")
def day_demand_predict_train(prediction_date: prediction_date):
    training_model = DayDemandPredictionModel(DATASET_PATH, prediction_date.date, PREDICTION_PATH)
    trained_model = training_model.predict_date()
    return {"model"+""+" trained on data before requested date :)"}

@app.post("/get_day_demand")
def get_day_demand(prediction_location: prediction_date_location):
    get_demand_obj = GetDemand(PREDICTION_PATH,prediction_location)
    predicted_demand = get_demand_obj.get_location_demand()
    return predicted_demand



