
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from typing import Union

from training_models.demand_prediction_train import Demand_Prediction_Train_Model
from training_models.get_demand import Get_Demand

class prediction_date(BaseModel):
    date: str
    interval: Union[int, None] = None

class prediction_location(BaseModel):
    date: str
    interval: Union[int, None] = None
    location_ids: list[int]

# pydantic handle date input
# pydantic handle location in range
app = FastAPI()

PREDICTION_PATH = '/home/saadi/DS/shoofer_demand_deploy/data/predictions/'
DATASET_PATH = '/home/saadi/DS/shoofer_demand_deploy/data/datasets'

@app.get("/")
def root():                                                                                               
    return {"API Provided": {
        "train model":"/train_demand_predict",
        "get predicted demand":"/get_demand"
    }}

@app.post("/train_demand_predict")
def demand_predict_train(prediction_date: prediction_date):
    training_model = Demand_Prediction_Train_Model(DATASET_PATH, prediction_date.date, PREDICTION_PATH)
    trained_model = training_model.predict_model()
    return {"model"+""+" trained on data before requested date :)"}

@app.post("/get_demand")
def get_demand(prediction_location: prediction_location):
    get_demand_obj = Get_Demand(PREDICTION_PATH+"xgb_predicted.parquet")
    predicted_demand = get_demand_obj.get_location_demand(prediction_location)
    return predicted_demand



