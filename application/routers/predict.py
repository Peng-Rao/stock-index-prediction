from datetime import datetime, timedelta

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
    "5_step_LSTM": rt.InferenceSession(models["LSTM-7-5"]),
    "15_step_LSTM": rt.InferenceSession(models["LSTM-7-15"]),
    "30_step_LSTM": rt.InferenceSession(models["LSTM-7-30"]),
    "5_step_Transformer": rt.InferenceSession(models["Transformer-7-5"]),
    "15_step_Transformer": rt.InferenceSession(models["Transformer-7-15"]),
    "30_step_Transformer": rt.InferenceSession(models["Transformer-7-30"]),
    "5_step_LSTM-Transformer": rt.InferenceSession(models["LSTM-Transformer-7-5"]),
    "15_step_LSTM-Transformer": rt.InferenceSession(models["LSTM-Transformer-7-15"]),
    "30_step_LSTM-Transformer": rt.InferenceSession(models["LSTM-Transformer-7-30"]),
}

n_lags = 7


class TimeSeriesSequence(BaseModel):
    sequence: list[list[float]]


class PredictRequest(BaseModel):
    # 股票代码，默认为 sh000001
    symbol: str = "sh000001"
    # 选取的模型，默认为 LSTM-Transformer
    model: str = "LSTM-Transformer"
    # 预测步数，默认为 5
    step: int = 5


router = APIRouter()


@router.post("/predict")
def predict(request: PredictRequest, db: Session = Depends(get_mysql_session)):
    # 获取模型和标准化器
    model = models[f"{request.step}_step_{request.model}"]
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
    # 输入模型
    input_name = model.get_inputs()[0].name
    # 输出模型
    output_name = model.get_outputs()[0].name
    # 预测
    prediction = model.run([output_name], {input_name: sequence.astype(np.float32)})[0]
    # 反归一化
    prediction = y_scaler.inverse_transform(prediction)

    # 获取未来 5 天，并转化为2024-05-01格式字符串
    future_dates = [
        (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(1, request.step + 1)
    ]
    prediction = prediction.flatten().tolist()
    return [
        {"time": date, "value": prediction[i]} for i, date in enumerate(future_dates)
    ]
