import streamlit as st
import tensorflow as tf
import numpy as np

# Page configuration for a professional wide-screen white layout
st.set_page_config(
    page_title="CSAT Intelligence Hub", 
    layout="wide", 
    page_icon="📊"
)

# Custom CSS for Total White Background & "PhonePe" Style UI
st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF;
    }
    .metric-container {
        background-color: #fcfcfc;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #eeeeee;
    }
    .stButton>button {
        width: 100%;
        background-color: #5f259f; /* PhonePe Purple */
        color: white;
        border-radius: 8px;
        height: 3.5em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_my_model():
    try:
        # Rebuilding skeleton to bypass deserialization errors
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

# --- HEADER ---
st.title("📊 Customer Satisfaction (CSAT) Intelligence Hub")
st.write("Professional ANN Engine for Real-time Satisfaction Forecasting")
st.divider()

# --- INTERACTION METRICS (FRONT & CENTER) ---
st.subheader("📋 Interaction Metrics")
st.info("Adjust the 12 Key Performance Indicators (KPIs) below for analysis.")

# Organizing into 3 columns for a clean center-screen look
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
    f12 = st.slider("Overall Service Quality", 0.0, 1.0, 0.5)

st.divider()

# --- PREDICTION ---
res_col1, res_col2 = st.columns([1, 1])

with res_col1:
    if st.button("Generate Analytics Score"):
        if model:
            input_data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]], dtype=np.float32)
            prediction = model.predict(input_data)
            score = np.argmax(prediction) + 1
            
            st.metric(label="Predicted CSAT Score", value=f"{score} / 5 ⭐")
            
            if score >= 4: st.success("Outcome: High Customer Satisfaction")
            elif score == 3: st.warning("Outcome: Neutral Customer Experience")
            else: st.error("Outcome: Dissatisfaction Risk Detected")
        else:
            st.error("System Offline: Model weights not found.")

with res_col2:
    st.markdown("### System Insights")
    st.write(f"**Model Type:** Artificial Neural Network (ANN)")
    st.write(f"**Input Shape:** 12 Features")
    st.write("**Status:** ✅ Operational")
