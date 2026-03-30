import streamlit as st
import tensorflow as tf
import numpy as np

# Page configuration for a professional wide-screen white layout
st.set_page_config(
    page_title="CSAT Intelligence Hub", 
    layout="wide", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS for the "PhonePe" White Theme & Card Styling
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #5f259f; /* PhonePe Purple tone */
        color: white;
    }
    .status-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f7ff;
        border-left: 5px solid #007bff;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_my_model():
    try:
        # Rebuilding skeleton to bypass the quantization_config/deserialization errors
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

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=80)
    st.header("Interaction Metrics")
    st.write("Adjust KPIs for analysis:")
    
    # Using sliders for a professional feel
    f1 = st.slider("Response Time", 0.0, 1.0, 0.5)
    f2 = st.slider("Agent Behavior", 0.0, 1.0, 0.5)
    f3 = st.slider("Issue Resolution", 0.0, 1.0, 0.5)
    f4 = st.slider("Communication Quality", 0.0, 1.0, 0.5)
    f5 = st.slider("Service Speed", 0.0, 1.0, 0.5)
    f6 = st.slider("Customer Effort", 0.0, 1.0, 0.5)
    f7 = st.slider("Product Knowledge", 0.0, 1.0, 0.5)
    f8 = st.slider("Problem Understanding", 0.0, 1.0, 0.5)
    f9 = st.slider("Follow-up Support", 0.0, 1.0, 0.5)
    f10 = st.slider("Courtesy Level", 0.0, 1.0, 0.5)
    f11 = st.slider("Waiting Time", 0.0, 1.0, 0.5)
    f12 = st.slider("Service Quality", 0.0, 1.0, 0.5)

# --- MAIN CONTENT ---
st.title("📊 CSAT Prediction Intelligence Hub")
st.markdown("##### Strategic Customer Satisfaction Analytics Engine")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("""
        <div class="status-card">
            <h3>System Status: Active</h3>
            <p>Our Deep Learning Artificial Neural Network (ANN) is currently processing 12 real-time 
            Key Performance Indicators to forecast customer sentiment accurately.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Adding a visual guide for the user
    st.subheader("Analysis Insights")
    st.write("""
        * **Accuracy:** The model uses weights optimized during training in Colab.
        * **Input:** 12 specific features matching your `csat_model.keras` architecture.
        * **Target:** Predicting classes 1 through 5 for direct CSAT scoring.
    """)

with col2:
    st.subheader("Prediction Console")
    if st.button("Generate Score"):
        if model:
            input_data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]], dtype=np.float32)
            prediction = model.predict(input_data)
            score = np.argmax(prediction) + 1
            
            # Display score in a big "PhonePe" style metric
            st.metric(label="Predicted CSAT Score", value=f"{score} / 5 ⭐")
            
            # Contextual Feedback
            if score >= 4:
                st.success("Exceeding Expectations")
            elif score == 3:
                st.warning("Average Performance")
            else:
                st.error("Immediate Attention Required")
        else:
            st.error("Model Error: Please verify weights in GitHub.")

st.divider()
st.caption("AI/ML Internship Project | Analytics Powered by TensorFlow & Streamlit")
