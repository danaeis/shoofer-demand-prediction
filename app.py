
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from typing import Union

from training_models.demand_prediction import Demand_Prediction_Model
from training_models.get_demand import Get_Demand

class prediction_date(BaseModel):
    date: str
    interval: Union[int, None] = None

class prediction_location(BaseModel):
    date: str
    interval: Union[int, None] = None
    location_ids: list[int]

# list of locations
    # add range

# pydantic handle date input

app = FastAPI()

PREDICTION_PATH = '/home/saadi/DS/shoofer_demand_deploy/data/predictions/predicted.parquet'
DATASET_PATH = '/home/saadi/DS/shoofer_demand_deploy/data/datasets'

@app.get("/")
def root():                                                                                               
    return {"message": "Hello World"}
# pass path as argoman
# change api name
@app.post("/demand_predict")
def demand_predict(prediction_date: prediction_date):
    training_model = Demand_Prediction_Model(DATASET_PATH, prediction_date.date, PREDICTION_PATH)
    trained_model = training_model.predict_model()
    # write a message
    return {"model trained on data before requested date :)"}

@app.post("/get_demand")
def get_demand(prediction_location: prediction_location):
    get_demand_obj = Get_Demand(PREDICTION_PATH)
    predicted_demand = get_demand_obj.get_location_demand(prediction_location)
    return predicted_demand



