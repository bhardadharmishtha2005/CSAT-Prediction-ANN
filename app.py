import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

# 1. Rebuild the Model Architecture (Must match your training code exactly)
def load_trained_model():
    model = Sequential([
        # Change 'units' to the best one found by your Tuner (e.g., 64 or 128)
        Dense(64, activation='relu', input_shape=(1,)), 
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(5, activation='softmax')
    ])
    
    # 2. Load the weights from your .keras file
    # If this fails, make sure the file name is exactly 'csat_model.keras'
    model.load_weights('csat_model.keras')
    return model

model = load_trained_model()

st.title("📊 CSAT Prediction Dashboard")

# 3. Simple Input
res_time = st.number_input("Resolution Time (Minutes)", min_value=0)

if st.button("Predict"):
    features = np.array([[res_time]])
    prediction = model.predict(features)
    score = np.argmax(prediction) + 1
    st.success(f"Predicted CSAT Score: {score} ⭐")
