from functools import lru_cache

from pyrate_limiter import Duration, Limiter, RequestRate
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket


class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass


@lru_cache
def get_yfinance_session():
    session = CachedLimiterSession(
        limiter=Limiter(
            RequestRate(5, Duration.SECOND * 5)
        ),  # max 2 requests per 5 seconds
        bucket_class=MemoryQueueBucket,
        backend=SQLiteCache("yfinance.cache"),
    )
    return session
