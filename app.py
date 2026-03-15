import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

@st.cache_resource
def load_my_model():
    try:
        # Loading with compile=False to bypass version-specific optimizer errors
        return tf.keras.models.load_model('csat_model.keras', compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

# Load your scaler if you used one (highly recommended for ANNs)
# scaler = joblib.load('scaler.joblib') 

st.title("📊 CSAT Prediction Dashboard")

# You need 12 inputs to match the model's training data
col1, col2 = st.columns(2)

with col1:
    res_time = st.number_input("Resolution Time (Min)", min_value=0, value=10)
    feat2 = st.number_input("Feature 2", value=0)
    feat3 = st.number_input("Feature 3", value=0)
    feat4 = st.number_input("Feature 4", value=0)
    feat5 = st.number_input("Feature 5", value=0)
    feat6 = st.number_input("Feature 6", value=0)

with col2:
    feat7 = st.number_input("Feature 7", value=0)
    feat8 = st.number_input("Feature 8", value=0)
    feat9 = st.number_input("Feature 9", value=0)
    feat10 = st.number_input("Feature 10", value=0)
    feat11 = st.number_input("Feature 11", value=0)
    feat12 = st.number_input("Feature 12", value=0)

if st.button("Predict"):
    if model is not None:
        # Create an array with all 12 features
        input_data = np.array([[res_time, feat2, feat3, feat4, feat5, feat6, 
                                feat7, feat8, feat9, feat10, feat11, feat12]], dtype=np.float32)
        
        # If you used a scaler in Colab, apply it here:
        # input_data = scaler.transform(input_data)
        
        prediction = model.predict(input_data)
        score = np.argmax(prediction) + 1
        st.success(f"Predicted CSAT Score: {score} ⭐")
