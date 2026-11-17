
import streamlit as st
import numpy as np
import tensorflow as tf

st.set_page_config(
    page_title="Time Series Prediction",
    page_icon="📈"
)

st.title("Time Series Prediction")
st.write("Predict the next value using RNN, LSTM and GRU.")

SCALE = 500.0

@st.cache_resource
def load_models():
    rnn = tf.keras.models.load_model("rnn_model.keras")
    lstm = tf.keras.models.load_model("lstm_model.keras")
    gru = tf.keras.models.load_model("gru_model.keras")

    return rnn, lstm, gru

try:
    rnn_model, lstm_model, gru_model = load_models()

    st.success("All three models loaded successfully!")

    st.subheader("Enter Previous Three Values")

    user_input = st.text_input(
        "Enter values separated by commas:",
        "30,40,50"
    )

    if st.button("Predict Next Value"):

        try:
            values = [
                float(x.strip())
                for x in user_input.split(",")
            ]

            if len(values) != 3:
                st.error("Please enter exactly three values.")

            else:
                input_data = np.array(
                    values,
                    dtype=np.float32
                ) / SCALE

                input_data = input_data.reshape(
                    1, 3, 1
                )

                rnn_prediction = (
                    rnn_model.predict(input_data, verbose=0)[0][0] * SCALE
                )

                lstm_prediction = (
                    lstm_model.predict(input_data, verbose=0)[0][0] * SCALE
                )

                gru_prediction = (
                    gru_model.predict(input_data, verbose=0)[0][0] * SCALE
                )

                st.subheader("Prediction Results")

                st.metric("RNN Prediction", f"{rnn_prediction:.2f}")
                st.metric("LSTM Prediction", f"{lstm_prediction:.2f}")
                st.metric("GRU Prediction", f"{gru_prediction:.2f}")

        except ValueError:
            st.error("Enter valid numbers separated by commas.")

except Exception as e:
    st.error(f"Model loading error: {e}")
