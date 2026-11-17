
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import numpy as np
import tensorflow as tf

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
SCALE = 500.0

models = None

def load_models():
    global models

    if models is None:
        models = {
            "RNN": tf.keras.models.load_model(
                BASE_DIR / "rnn_model.keras"
            ),
            "LSTM": tf.keras.models.load_model(
                BASE_DIR / "lstm_model.keras"
            ),
            "GRU": tf.keras.models.load_model(
                BASE_DIR / "gru_model.keras"
            )
        }

    return models


class PredictionInput(BaseModel):
    values: list[float] = Field(min_length=3, max_length=3)


@app.get("/api/health")
def health():
    return {"status": "API is running"}


@app.post("/api/predict")
def predict(data: PredictionInput):

    try:
        loaded_models = load_models()

        input_data = np.array(
            data.values, dtype=np.float32
        ) / SCALE

        input_data = input_data.reshape(1, 3, 1)

        results = {}

        for name, model in loaded_models.items():
            prediction = model.predict(
                input_data, verbose=0
            )[0][0]

            results[name] = round(
                float(prediction * SCALE), 2
            )

        return {
            "input": data.values,
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
