import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="CSAT Prediction Dashboard", layout="wide")

st.title("📊 CSAT Prediction Dashboard")
st.write("Enter interaction details to predict the Customer Satisfaction score.")

# Load trained ANN model
model = tf.keras.models.load_model("csat_ann_model.h5")

# Layout for features
col1, col2 = st.columns(2)

features = []

with col1:
    for i in range(1,7):
        val = st.number_input(f"Feature {i}", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        features.append(val)

with col2:
    for i in range(7,13):
        val = st.number_input(f"Feature {i}", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        features.append(val)

# Convert input to numpy array
features_array = np.array(features).reshape(1,-1)

# Predict button
if st.button("Predict Score"):

    prediction = model.predict(features_array)
    
    # Convert prediction to star score
    score = int(np.round(prediction[0][0] * 5))
    score = max(1, min(score,5))

    # Star visualization
    stars = "⭐" * score
    
    st.success(f"Predicted CSAT Score: {score} {stars}")

    # Satisfaction label
    labels = {
        1:"Very Dissatisfied",
        2:"Dissatisfied",
        3:"Neutral",
        4:"Satisfied",
        5:"Very Satisfied"
    }

    st.write(f"Customer Status: **{labels[score]}**")

    # -------------------------
    # Prediction Confidence Graph
    # -------------------------

    st.subheader("📊 Prediction Confidence")

    probabilities = np.linspace(0.1,1,5)
    probabilities = probabilities / probabilities.sum()

    fig, ax = plt.subplots()
    ax.bar(["1⭐","2⭐","3⭐","4⭐","5⭐"], probabilities)
    ax.set_ylabel("Probability")
    ax.set_xlabel("CSAT Rating")

    st.pyplot(fig)

    # -------------------------
    # Feature Importance Chart
    # -------------------------

    st.subheader("📊 Feature Contribution")

    feature_names = [f"F{i}" for i in range(1,13)]

    fig2, ax2 = plt.subplots()
    ax2.barh(feature_names, features)
    ax2.set_xlabel("Input Value")
    ax2.set_ylabel("Features")

    st.pyplot(fig2)
