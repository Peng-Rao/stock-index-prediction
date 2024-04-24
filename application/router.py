from fastapi import APIRouter, Depends
from predict import predict
from datamodel import TimeSeriesFeatures, PredictedResult

router = APIRouter()


@router.post("/predict", response_model=PredictedResult)
def post_predict(timeseries: TimeSeriesFeatures,):
    return predict(timeseries)
