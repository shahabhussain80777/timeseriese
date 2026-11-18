from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Time Series Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "OK"}
