import time
from typing import List

import akshare as ak
import pandas as pd
from dependencies import get_db_engine
from models import StockIndex, StockIndexHistory
from sqlmodel import Session, SQLModel, select

db_engine = get_db_engine()
SQLModel.metadata.create_all(db_engine)


def df_to_sqlmodel(df: pd.DataFrame, model: SQLModel) -> List[SQLModel]:
    """Convert a pandas DataFrame into a a list of SQLModel objects."""
    objs = [model(**row) for row in df.to_dict("records")]
    return objs


def get_stock_index_list() -> None:
    stock_df = ak.stock_zh_index_spot_sina()
    stock_df = stock_df[["代码", "名称"]]
    stock_df.rename(columns={"代码": "symbol", "名称": "name"}, inplace=True)
    with Session(db_engine) as session:
        stock_index_list = df_to_sqlmodel(stock_df, StockIndex)
        session.add_all(stock_index_list)
        session.commit()


def get_stock_index_history(symbol: str) -> None:
    stock_df = ak.stock_zh_index_daily_em(symbol=symbol)
    stock_df["symbol"] = symbol
    with Session(db_engine) as session:
        stock_index_list = df_to_sqlmodel(stock_df, StockIndexHistory)
        session.add_all(stock_index_list)
        session.commit()


if __name__ == "__main__":
    # 获取股票指数列表
    with Session(db_engine) as session:
        statement = select(StockIndex)
        stock_index = session.exec(statement).all()
        symbol_list = [stock_index.symbol for stock_index in stock_index]
        for stock_index in symbol_list:
            get_stock_index_history(stock_index)
            time.sleep(5)
