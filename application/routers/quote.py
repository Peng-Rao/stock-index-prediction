from typing import Dict, List, Union

import yfinance as yf
from dependencies import get_yfinance_session
from fastapi import APIRouter, Depends, HTTPException, Query

router = APIRouter()


@router.get("/quote/", tags=["quote"])
async def get_quotes(
    symbols: List[str] = Query(None), session=Depends(get_yfinance_session)
) -> Dict[str, Union[str, List[dict]]]:
    if not symbols:
        raise HTTPException(status_code=400, detail="No symbols provided.")

    data: List[dict] = []
    error_messages: List[str] = []

    if len(symbols) == 1:
        symbols = symbols[0].split(",")

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol, session=session)
            info = ticker.info  # Fetch ticker info
            history_day = ticker.history(period="1d")
            history_3mo = ticker.history(period="3mo")

            quote = {
                "currency": info.get("currency"),
                "marketState": "CLOSE",
                "fullExchangeName": info.get("exchange"),
                "displayName": info.get("longName"),
                "symbol": info.get("symbol"),
                "regularMarketPrice": info.get("currentPrice"),
                "regularMarketChange": history_day.iloc[-1]["Close"]
                - history_day.iloc[0]["Close"],
                "regularMarketChangePercent": (
                    (history_day.iloc[-1]["Close"] - history_day.iloc[0]["Close"])
                    / history_day.iloc[0]["Close"]
                )
                * 100,
                "RegularMarketChangePreviousClose": info.get(
                    "regularMarketPreviousClose"
                )
                - history_day.iloc[0]["Close"],
                "postMarketPrice": history_day.iloc[-1]["Close"],
                "postMarketPriceChange": history_day.iloc[-1]["Close"]
                - history_day.iloc[0]["Close"],
                "regularMarketOpen": info.get("regularMarketOpen"),
                "regularMarketDayHigh": info.get("regularMarketDayHigh"),
                "regularMarketDayLow": info.get("regularMarketDayLow"),
                "regularMarketVolume": info.get("regularMarketVolume"),
                "trailingPE": info.get("trailingPE"),
                "marketCap": info.get("marketCap"),
                "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
                "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
                "averageDailyVolume3Month": history_3mo.Volume.mean(),
                "trailingAnnualDividendYield": info.get("trailingAnnualDividendYield"),
                "epsTrailingTwelveMonths": info.get("trailingEps"),
            }
            data.append(quote)

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    if error_messages:
        return {"quotes": data, "error": error_messages}
    return {"quotes": data, "error": "None"}
