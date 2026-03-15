import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import numpy as np

# Use the rebuild logic to bypass metadata errors
@st.cache_resource
def load_my_model():
    try:
        # 1. Build the skeleton (Must match your Colab architecture)
        model = Sequential([
            Input(shape=(12,)), 
            Dense(96, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(5, activation='softmax')
        ])
        # 2. Load only the weights
        model.load_weights('csat_model.keras')
        return model
    except Exception as e:
        st.error(f"Load Error: {e}")
        return None

model = load_my_model()

st.title("📊 CSAT Prediction Dashboard")

# Create 12 inputs to match the model's expected shape [None, 12]
inputs = []
cols = st.columns(2)
for i in range(12):
    with cols[i % 2]:
        val = st.number_input(f"Feature {i+1}", value=0.0)
        inputs.append(val)

if st.button("Predict"):
    if model:
        # Convert list to numpy array with shape (1, 12)
        data = np.array([inputs], dtype=np.float32)
        prediction = model.predict(data)
        # argmax + 1 converts 0-4 index to 1-5 score
        result = np.argmax(prediction) + 1 
        st.success(f"Predicted CSAT Score: {result} ⭐")
