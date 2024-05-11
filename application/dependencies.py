from functools import lru_cache

from pyrate_limiter import Duration, Limiter, RequestRate
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from sqlmodel import SQLModel, create_engine


class MySQLSession:
    MYSQL_URL = "mysql+pymysql://root:123456@localhost:3306/A_stock?charset=utf8"
    POOL_SIZE = 20
    POOL_RECYCLE = 3600
    POOL_TIMEOUT = 15
    MAX_OVERFLOW = 2
    CONNECT_TIMEOUT = 60

    @classmethod
    def get_db_engine(cls):
        engine = create_engine(
            cls.MYSQL_URL,
            pool_size=cls.POOL_SIZE,
            pool_recycle=cls.POOL_RECYCLE,
            pool_timeout=cls.POOL_TIMEOUT,
            max_overflow=cls.MAX_OVERFLOW,
            echo=True,
        )
        return engine

    @classmethod
    def get_db_session(cls) -> Session:
        try:
            engine = cls.get_db_engine()
            SQLModel.metadata.create_all(engine)
            return Session(engine)
        except Exception as e:
            print("Error getting DB session:", e)
            return None


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
def get_db_engine() -> Session:
    return MySQLSession.get_db_engine()


@lru_cache
def get_db_session() -> Session:
    return MySQLSession.get_db_session()