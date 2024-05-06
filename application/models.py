from datetime import datetime

from sqlmodel import Field, SQLModel


class StockIndex(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    symbol: str
    name: str


class StockIndexHistory(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    date: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    amount: float


class StockIndexRealtime(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    # 股票代码
    symbol: str
    # 股票名称
    name: str
    # 当前价格
    current_price: float
    # 涨跌
    regular_market_change: float
    # 涨跌幅
    regular_market_change_percent: float
    # 昨收
    regular_market_previous_close: float
    # 今开
    regular_market_open: float
    # 最高
    regular_market_day_high: float
    # 最低
    regular_market_day_low: float
    # 成交量
    regular_market_volume: int
    # 成交额
    regular_market_turnover: float
    # 时间
    created_at: datetime = Field(default_factory=datetime.now)
