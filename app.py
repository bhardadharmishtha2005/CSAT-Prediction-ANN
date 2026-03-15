import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

# Load the saved model
model = tf.keras.models.load_model('csat_model.keras')

st.title("📊 CSAT Prediction Dashboard")
st.write("Enter the details below to predict Customer Satisfaction.")

# Simple Inputs
res_time = st.number_input("Resolution Time (Minutes)", min_value=1, max_value=1000)
category = st.selectbox("Category", ['Returns', 'Cancellations', 'General Query', 'Payment'])

if st.button("Predict Score"):
    # Note: Ensure your input matches the shape of X_test (your features)
    # This is a simplified example; use your actual feature processing here
    prediction = model.predict(np.array([[res_time]])) 
    score = np.argmax(prediction) + 1
    st.success(f"Predicted CSAT Score: {score} ⭐")
