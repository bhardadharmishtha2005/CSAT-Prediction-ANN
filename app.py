import streamlit as st
import tensorflow as tf
import numpy as np

# Page configuration for a clean, white, wide-screen layout
st.set_page_config(
    page_title="CSAT Analytics Hub", 
    layout="wide", 
    page_icon="📊"
)

# CSS for Total White Background, Card Styling, and Purple Accents
st.markdown("""
    <style>
    /* Force total white background */
    .stApp {
        background-color: #FFFFFF;
    }
    /* Hide the sidebar for a centered experience */
    [data-testid="stSidebar"] {
        display: none;
    }
    /* Metric and Input Card Styling */
    .metric-container {
        background-color: #fcfcfc;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #eeeeee;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }
    .stSlider > div [data-baseweb="slider"] {
        width: 95%;
    }
    /* Professional Purple Button (PhonePe style) */
    .stButton>button {
        background-color: #5f259f;
        color: white;
        border-radius: 8px;
        height: 3.5em;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #4b1d7d;
        border-color: #4b1d7d;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_my_model():
    try:
        # Rebuilding the architecture to match your 12-feature training
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
        return None

model = load_my_model()

# --- HEADER SECTION ---
st.title("📊 CSAT Prediction Intelligence Hub")
st.write("Professional Deep Learning Engine for Customer Satisfaction Forecasting")
st.divider()

# --- INTERACTION METRICS (FRONT & CENTER) ---
st.subheader("📋 Interaction Metrics")
st.info("Adjust the sliders below to represent the customer service interaction parameters.")

# Create 3 columns to organize the 12 features professionally
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    f1 = st.slider("Response Time", 0.0, 1.0, 0.5)
    f2 = st.slider("Agent Behavior", 0.0, 1.0, 0.5)
    f3 = st.slider("Issue Resolution", 0.0, 1.0, 0.5)
    f4 = st.slider("Communication Quality", 0.0, 1.0, 0.5)

with c2:
    f5 = st.slider("Service Speed", 0.0, 1.0, 0.5)
    f6 = st.slider("Customer Effort", 0.0, 1.0, 0.5)
    f7 = st.slider("Product Knowledge", 0.0, 1.0, 0.5)
    f8 = st.slider("Problem Understanding", 0.0, 1.0, 0.5)

with c3:
    f9 = st.slider("Follow-up Support", 0.0, 1.0, 0.5)
    f10 = st.slider("Courtesy Level", 0.0, 1.0, 0.5)
    f11 = st.slider("Waiting Time", 0.0, 1.0, 0.5)
    f12 = st.slider("Service Quality", 0.0, 1.0, 0.5)

st.markdown("---")

# --- PREDICTION & RESULTS SECTION ---
res_col1, res_col2 = st.columns([1, 1])

with res_col1:
    st.subheader("Model Prediction")
    if st.button("Generate Analytics Report"):
        if model:
            input_data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]], dtype=np.float32)
            prediction = model.predict(input_data)
            score = np.argmax(prediction) + 1
            
            # Big Metric Output
            st.metric(label="Predicted CSAT Score", value=f"{score} / 5 ⭐")
            
            if score >= 4:
                st.success("Result: High Customer Satisfaction")
            elif score == 3:
                st.warning("Result: Neutral Customer Experience")
            else:
                st.error("Result: High Dissatisfaction Risk")
        else:
            st.error("System Error: Weights file 'csat_model.keras' not detected.")

with res_col2:
    st.subheader("System Insights")
    st.write(f"**Model Type:** Artificial Neural Network (ANN)")
    st.write(f"**Input Shape:** 12 Parallel Features")
    st.write(f"**Target:** Satisfaction Classification (1-5)")
    st.write("**Status:** ✅ Operational")

st.caption("AI/ML Internship | Developed by Bharda Dharmishtha | Labmentix 2026")
