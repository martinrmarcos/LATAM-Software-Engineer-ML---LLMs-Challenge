from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from fastapi import HTTPException
from model import DelayModel
from typing import List

app = FastAPI()
model = DelayModel()



class Flight(BaseModel):
    OPERA: str
    MES: int
    TIPOVUELO: str


class InputData(BaseModel):
    flights: List[Flight]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    if model._model is None:
        raise HTTPException(status_code=400, detail="Model not trained")
    else:
        return {
            "status": "ALL_OK"
        }

@app.get("/train", status_code=200)
async def get_health() -> dict:
    try:
        data = pd.read_csv('../data/data.csv')
        print('Dataset Fetched')
        data = model.preprocess(data, 'delay')
        model.fit(data[0], data[1])
        return {
            "status": "Model Trained"
        }
    except:
        raise HTTPException(status_code=400, detail="Model not trained")



@app.post("/predict/", status_code=200)
async def post_predict(data: InputData) -> dict:
    if model._model is None:
        raise HTTPException(status_code=400, detail="Model not trained")
    else:
        mes = pd.json_normalize(data.dict(), record_path='flights')['MES'].values[0]
        print(mes)
        if mes <= 12:
            data = pd.json_normalize(data.dict(), record_path='flights')
            data = model.preprocess(data)
            prediction = model.predict(data)

            return {
                "status": "OK",
                "predict": prediction
            }
        else:
            raise HTTPException(status_code=400, detail="Month value over 12")
