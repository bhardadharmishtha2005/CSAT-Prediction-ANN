import streamlit as st
import tensorflow as tf
import numpy as np

# 1. Load the model
# Using compile=False helps avoid version errors during loading
model = tf.keras.models.load_model('csat_model.keras', compile=False)

st.title("📊 CSAT Prediction Dashboard")
st.write("Enter details to predict Customer Satisfaction score (1-5).")

# 2. Setup Inputs (Adjust these based on your actual X features)
res_time = st.number_input("Resolution Time (Minutes)", min_value=0)
# Add other inputs here if you have more features!

if st.button("Predict"):
    # Reshape input to match what the model expects
    # If you have 10 features, change '1' to '10'
    features = np.array([[res_time]]) 
    
    prediction = model.predict(features)
    score = np.argmax(prediction) + 1
    
    st.success(f"The predicted CSAT score is: {score} ⭐")
