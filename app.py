
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from typing import Union

from training_models.demand_prediction import Demand_Prediction_Model

class prediction_date(BaseModel):
    date: str
    interval: Union[int, None] = None

app = FastAPI()

@app.get("/")
def root():                                                                                               
    return {"message": "Hello World"}

@app.post("/demand_predict")
def demand_predict(prediction_date: prediction_date):
    training_model = Demand_Prediction_Model(prediction_date.date)
    trained_model = training_model.check()
    return trained_model
    predicts = trained_model

