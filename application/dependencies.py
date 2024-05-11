from functools import lru_cache
from pathlib import Path

from pyrate_limiter import Duration, Limiter, RequestRate
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from sqlmodel import Session as db_session
from sqlmodel import create_engine

MYSQL_DATABASE_URL = "mysql+pymysql://root:123456@localhost:3306/A_stock?charset=utf8"
engine = create_engine(MYSQL_DATABASE_URL, echo=True)


def get_mysql_session():
    with db_session(engine) as session:
        yield session


class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass


@lru_cache
def get_yfinance_session():
    session = CachedLimiterSession(
        limiter=Limiter(
            RequestRate(20, Duration.SECOND * 5)
        ),  # max 20 requests per 5 seconds
        bucket_class=MemoryQueueBucket,
        backend=SQLiteCache("yfinance.cache"),
    )
    return session


@lru_cache
def get_sclaers():
    # 获取 sclaer 目录下的所有文件
    scalers_dir = Path(__file__).parent / "scalers"
    scalers = Path(scalers_dir).rglob("*.pkl")
    scalers_dict = {}
    for scaler in scalers:
        scalers_dict[scaler.stem] = scaler
    return scalers_dict


@lru_cache
def get_deep_models():
    models_dir = Path(__file__).parent / "lib" / "models"
    models = Path(models_dir).rglob("*.onnx")
    models_dict = {}
    for model in models:
        models_dict[model.stem] = model
    return models_dict
