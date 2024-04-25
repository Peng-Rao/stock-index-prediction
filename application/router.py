from datamodel import PredictedResult, TimeSeriesFeatures
from fastapi import APIRouter, Depends
from predict import predict

router = APIRouter()


@router.post("/predict", response_model=PredictedResult)
def post_predict(timeseries: TimeSeriesFeatures,):
    return predict(timeseries)
