import onnxruntime as rt
from fastapi import APIRouter
from pydantic import BaseModel


class TimeSeriesSequence(BaseModel):
    sequence: list[list[float]]


router = APIRouter()
# session = rt.InferenceSession(MODEL_PATH)
# input_name = session.get_inputs()[0].name
# label_name = session.get_outputs()[0].name


# def predict(data: TimeSeriesFeatures) -> PredictedResult:
#     predicted = session.run(
#         output_names=[label_name], input_feed={input_name: data.to_numpy()}
#     )
#     return PredictedResult(
#         **{"predicted": PredictedResult.transform(predicted[0][0][0])}
#     )


# @router.post("/predict", response_model=PredictedResult)
# def post_predict(
#     timeseries: TimeSeriesFeatures,
# ):
#     return predict(timeseries)
