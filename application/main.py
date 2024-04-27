import uvicorn
from fastapi import FastAPI
from routers import predict, quote

app = FastAPI()
app.include_router(quote.router)
app.include_router(predict.router)

if __name__ == "__main__":
    uvicorn.run(
        app="main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug"
    )
