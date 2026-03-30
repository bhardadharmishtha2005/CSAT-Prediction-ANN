import streamlit as st
import tensorflow as tf
import numpy as np

# Page configuration
st.set_page_config(page_title="CSAT Analytics Hub", layout="wide")

# CSS to fix visibility: Force dark text on white background
st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF;
    }
    /* Force all text to be dark grey/black for visibility */
    h1, h2, h3, h4, h5, h6, p, label, .stSlider {
        color: #1c1c1c !important;
    }
    /* Style for the Interaction Metrics section */
    .metric-box {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #eeeeee;
    }
    /* PhonePe Purple Button */
    .stButton>button {
        background-color: #5f259f;
        color: white !important;
        width: 100%;
        border-radius: 8px;
        height: 3em;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_my_model():
    try:
        # Rebuilding skeleton to match your 12-feature training logs
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

# --- MAIN UI ---
st.title("📊 CSAT Prediction Intelligence Hub")
st.markdown("##### Professional Deep Learning Engine for Customer Satisfaction Forecasting")
st.divider()

st.subheader("📋 Interaction Metrics")
st.write("Adjust the sliders below to represent the customer service parameters.")

# Grid layout for 12 features (Front & Center)
c1, c2, c3 = st.columns(3)

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

st.divider()

# Prediction Area
col_left, col_right = st.columns([1, 1])

with col_left:
    if st.button("Generate Analytics Report"):
        if model:
            data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]], dtype=np.float32)
            prediction = model.predict(data)
            score = np.argmax(prediction) + 1
            st.metric(label="Predicted CSAT Score", value=f"{score} / 5 ⭐")
        else:
            st.error("Model file 'csat_model.keras' not found in repository.")

with col_right:
    st.info("**System Info:** ANN Architecture 12-96-32-5")
