from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as rt
import numpy as np

app = FastAPI()

# Load the three ONNX models
rnn_session = rt.InferenceSession("models/rnn_model.onnx")
gru_session = rt.InferenceSession("models/gru_model.onnx")
lstm_session = rt.InferenceSession("models/lstm_model.onnx")

SCALE = 500.0

class SequenceInput(BaseModel):
    numbers: list[float]

@app.post("/api/predict")
def predict(data: SequenceInput):
    if len(data.numbers) != 3:
        raise HTTPException(status_code=400, detail="Please enter exactly 3 numbers.")

    # Scale input and reshape to match the model requirement [1, 3, 1]
    seq = (np.array(data.numbers, dtype=np.float32) / SCALE).reshape(1, 3, 1)

    sessions = {"RNN": rnn_session, "LSTM": lstm_session, "GRU": gru_session}
    results = {}

    for name, session in sessions.items():
        input_name = session.get_inputs()[0].name
        pred_scaled = session.run(None, {input_name: seq})[0][0][0]
        results[name] = round(float(pred_scaled * SCALE), 2)

    return {"input": data.numbers, "predictions": results}

@app.get("/")
def home():
    return {"message": "Time Series API is running successfully!"}
