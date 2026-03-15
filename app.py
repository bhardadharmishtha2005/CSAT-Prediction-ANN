import streamlit as st
import tensorflow as tf
import numpy as np

# Use a decorator to cache the model so it doesn't reload on every click
@st.cache_resource
def load_my_model():
    # Attempt to load the full model first
    try:
        # 'compile=False' is key here to avoid version issues with optimizers
        return tf.keras.models.load_model('csat_model.keras', compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

st.set_page_config(page_title="CSAT Predictor", page_icon="📊")
st.title("📊 Customer Satisfaction (CSAT) Predictor")
st.write("This ANN model predicts satisfaction scores based on interaction data.")

# Sidebar for professional touch
st.sidebar.header("User Input")
res_time = st.sidebar.number_input("Resolution Time (Minutes)", min_value=0, value=10)

# Main Prediction Area
if model is not None:
    if st.button("Generate Prediction"):
        # Ensure the input shape is (1, number_of_features)
        # If your model was trained on more than 1 feature, 
        # you must provide all of them here in order.
        input_data = np.array([[res_time]], dtype=np.float32)
        
        prediction = model.predict(input_data)
        final_score = np.argmax(prediction) + 1
        
        # Display Result
        st.subheader("Result:")
        st.metric(label="Predicted CSAT Score", value=f"{final_score} / 5")
        
        if final_score >= 4:
            st.success("Great! The customer is likely satisfied. 😊")
        elif final_score == 3:
            st.warning("Neutral. There is room for improvement. 😐")
        else:
            st.error("Alert: Potential dissatisfied customer. 🚩")
else:
    st.warning("Model file not found. Please ensure 'csat_model.keras' is in your GitHub repository.")
