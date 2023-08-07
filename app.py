
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
    location_id: int

app = FastAPI()

@app.get("/")
def root():                                                                                               
    return {"message": "Hello World"}

@app.post("/demand_predict")
def demand_predict(prediction_date: prediction_date):
    training_model = Demand_Prediction_Model(prediction_date.date)
    trained_model = training_model.predict_model()
    return trained_model
    predicts = trained_model

@app.post("/get_demand")
def get_demand(prediction_date: prediction_date, prediction_location: prediction_location):
    get_demand_obj = Get_Demand()
    predicted_demand = get_demand_obj.get_location_demand(prediction_date, 
                                                          prediction_location.location_id)
    return predicted_demand



