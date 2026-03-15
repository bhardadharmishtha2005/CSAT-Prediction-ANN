import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import numpy as np

# Cache the model to avoid reloading on every click
@st.cache_resource
def load_my_model():
    try:
        # 1. Build the skeleton to match your Colab structure [12 inputs -> 96 -> 32 -> 5]
        model = Sequential([
            Input(shape=(12,)), 
            Dense(96, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(5, activation='softmax')
        ])
        
        # 2. Load only the weights (bypasses quantization_config errors)
        model.load_weights('csat_model.keras')
        return model
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

model = load_my_model()

st.title("📊 CSAT Prediction Dashboard")
st.write("Enter interaction details to predict the Customer Satisfaction score.")

# Create 12 input fields
inputs = []
col1, col2 = st.columns(2)
for i in range(12):
    with col1 if i < 6 else col2:
        val = st.number_input(f"Feature {i+1}", value=0.0, key=f"f{i}")
        inputs.append(val)

if st.button("Predict Score"):
    if model:
        # Convert to numpy array shape (1, 12)
        data = np.array([inputs], dtype=np.float32)
        prediction = model.predict(data)
        
        # Get score (1-5)
        score = np.argmax(prediction) + 1
        st.success(f"Predicted CSAT Score: {score} ⭐")
