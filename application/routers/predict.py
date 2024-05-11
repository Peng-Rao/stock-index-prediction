import joblib
import numpy as np
import onnxruntime as rt
from dependencies import get_deep_models, get_mysql_session, get_sclaers
from fastapi import APIRouter, Depends
from models import StockIndexHistory
from pydantic import BaseModel
from sqlmodel import Session, select

scalers = get_sclaers()
scalers = {scaler.stem: joblib.load(scaler) for scaler in scalers.values()}
models = get_deep_models()
models = {
    "1_step": rt.InferenceSession(models["LSTM-Transformer-7-1"]),
    "5_step": rt.InferenceSession(models["LSTM-Transformer-7-5"]),
    "30_step": rt.InferenceSession(models["LSTM-Transformer-90-30"]),
}

n_lags = 7


class TimeSeriesSequence(BaseModel):
    sequence: list[list[float]]


class PredictRequest(BaseModel):
    symbol: str  # 股票代码
    step: int  # 预测步数


router = APIRouter()


@router.post("/predict")
def predict(request: PredictRequest, db: Session = Depends(get_mysql_session)):
    # 获取模型和标准化器
    model = models[f"{request.step}_step"]
    X_scaler = scalers[request.symbol + "_X_scaler"]
    y_scaler = scalers[request.symbol + "_y_scaler"]
    # 获取数据，从数据库中获取最新的 step 个数据
    sql_statement = (
        select(StockIndexHistory)
        .where(StockIndexHistory.symbol == request.symbol)
        .order_by(StockIndexHistory.date.desc())
        .limit(n_lags)
    )
    results = db.exec(sql_statement)
    sequence = [
        [
            result.open,
            result.high,
            result.low,
            result.volume,
        ]
        for result in results
    ]
    # 转化为 numpy
    sequence = np.array(sequence).reshape(n_lags, 4)
    # 标准化
    sequence = X_scaler.transform(sequence)
    # 重塑
    sequence = sequence.reshape(1, n_lags, 4)
    # 预测
    # 输入模型
    input_name = model.get_inputs()[0].name
    # 输出模型
    output_name = model.get_outputs()[0].name
    # 预测
    prediction = model.run([output_name], {input_name: sequence.astype(np.float32)})[0]
    # 反归一化
    prediction = y_scaler.inverse_transform(prediction)
    return prediction.tolist()


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
