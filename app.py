import streamlit as st
import tensorflow as tf
import numpy as np

# Cache the model to prevent reloading on every interaction
@st.cache_resource
def load_my_model():
    try:
        # Loading with compile=False is essential to bypass version-specific optimizer errors
        return tf.keras.models.load_model('csat_model.keras', compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

st.set_page_config(page_title="CSAT Predictor", page_icon="📊")
st.title("📊 Customer Satisfaction (CSAT) Predictor")
st.write("Enter the 12 features from your dataset to predict the CSAT score.")

# Professional layout with two columns for the 12 inputs
col1, col2 = st.columns(2)

with col1:
    f1 = st.number_input("Feature 1 (e.g., Res Time)", value=0.0)
    f2 = st.number_input("Feature 2", value=0.0)
    f3 = st.number_input("Feature 3", value=0.0)
    f4 = st.number_input("Feature 4", value=0.0)
    f5 = st.number_input("Feature 5", value=0.0)
    f6 = st.number_input("Feature 6", value=0.0)

with col2:
    f7 = st.number_input("Feature 7", value=0.0)
    f8 = st.number_input("Feature 8", value=0.0)
    f9 = st.number_input("Feature 9", value=0.0)
    f10 = st.number_input("Feature 10", value=0.0)
    f11 = st.number_input("Feature 11", value=0.0)
    f12 = st.number_input("Feature 12", value=0.0)

if st.button("Generate Prediction"):
    if model is not None:
        # Combine all 12 inputs into the required (1, 12) shape
        input_data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]], dtype=np.float32)
        
        # Make the prediction
        prediction = model.predict(input_data)
        
        # Get the class with the highest probability (1-5)
        final_score = np.argmax(prediction) + 1
        
        st.subheader("Final Result:")
        st.success(f"The predicted CSAT score is: {final_score} ⭐")
    else:
        st.error("Model not loaded. Please check your GitHub repository for 'csat_model.keras'.")
