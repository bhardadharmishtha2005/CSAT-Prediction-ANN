import streamlit as st
import tensorflow as tf
import numpy as np

# Page config for a professional look
st.set_page_config(page_title="CSAT Intelligence Hub", layout="wide", page_icon="📊")

@st.cache_resource
def load_my_model():
    try:
        # Rebuilding skeleton to bypass quantization_config errors
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(12,)), 
            tf.keras.layers.Dense(96, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        model.load_weights('csat_model.keras')
        return model
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

model = load_my_model()

# --- SIDEBAR INPUTS (PhonePe Style) ---
st.sidebar.header("📋 Interaction Metrics")
st.sidebar.write("Adjust the values below:")

f1 = st.sidebar.slider("Response Time", 0.0, 1.0, 0.5)
f2 = st.sidebar.slider("Agent Behavior", 0.0, 1.0, 0.5)
f3 = st.sidebar.slider("Issue Resolution", 0.0, 1.0, 0.5)
f4 = st.sidebar.slider("Communication Quality", 0.0, 1.0, 0.5)
f5 = st.sidebar.slider("Service Speed", 0.0, 1.0, 0.5)
f6 = st.sidebar.slider("Customer Effort", 0.0, 1.0, 0.5)
f7 = st.sidebar.slider("Product Knowledge", 0.0, 1.0, 0.5)
f8 = st.sidebar.slider("Problem Understanding", 0.0, 1.0, 0.5)
f9 = st.sidebar.slider("Follow-up Support", 0.0, 1.0, 0.5)
f10 = st.sidebar.slider("Courtesy Level", 0.0, 1.0, 0.5)
f11 = st.sidebar.slider("Customer Waiting Time", 0.0, 1.0, 0.5)
f12 = st.sidebar.slider("Overall Service Quality", 0.0, 1.0, 0.5)

# --- MAIN DISPLAY ---
st.title("📊 CSAT Prediction Intelligence Hub")
st.markdown("---")

# Use columns for a clean look
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Model Status")
    if model:
        st.success("✅ DeepCSAT Model Active & Loaded")
    else:
        st.error("❌ Model Offline")

    st.info("This system uses an Artificial Neural Network (ANN) to analyze customer service efficiency based on 12 key performance indicators (KPIs).")

with col2:
    st.subheader("Predict Satisfaction")
    if st.button("Generate CSAT Score", use_container_width=True):
        if model:
            input_data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]], dtype=np.float32)
            prediction = model.predict(input_data)
            score = np.argmax(prediction) + 1
            
            # Big Metric display like PhonePe
            st.metric(label="Predicted Score", value=f"{score} / 5 ⭐")
            
            if score >= 4:
                st.balloons()
                st.success("High Satisfaction Predicted!")
            elif score == 3:
                st.warning("Neutral Sentiment Detected.")
            else:
                st.error("Dissatisfaction Risk Detected!")
