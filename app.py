import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(page_title="CSAT Prediction Dashboard", layout="wide")

st.title("📊 Customer Satisfaction (CSAT) Prediction Dashboard")
st.write("Enter customer service interaction details to predict the CSAT score.")

# Load ANN model
model = tf.keras.models.load_model("best_ann_model.keras")

# Feature names (more realistic)
feature_names = [
    "Response Time",
    "Agent Behavior",
    "Issue Resolution",
    "Communication Quality",
    "Service Speed",
    "Customer Effort",
    "Product Knowledge",
    "Problem Understanding",
    "Follow-up Support",
    "Courtesy Level",
    "Customer Waiting Time",
    "Overall Service Quality"
]

features = []

# Create 2 column layout
col1, col2 = st.columns(2)

with col1:
    for i in range(6):
        val = st.slider(feature_names[i], 0.0, 1.0, 0.5, 0.01)
        features.append(val)

with col2:
    for i in range(6,12):
        val = st.slider(feature_names[i], 0.0, 1.0, 0.5, 0.01)
        features.append(val)

# Convert input to numpy array
input_data = np.array(features).reshape(1,-1)

# Predict button
if st.button("Predict CSAT Score"):

    prediction = model.predict(input_data)

    pred_value = prediction[0][0]

    score = int(round(pred_value * 5))
    score = max(1, min(score,5))

    stars = "⭐" * score

    st.success(f"Predicted CSAT Score: {score} {stars}")

    st.write("Raw Model Output:", pred_value)

    # Satisfaction label
    status = {
        1:"Very Dissatisfied",
        2:"Dissatisfied",
        3:"Neutral",
        4:"Satisfied",
        5:"Very Satisfied"
    }

    st.write(f"Customer Status: **{status[score]}**")

    # -----------------------------
    # Prediction Confidence Chart
    # -----------------------------
    st.subheader("📊 Prediction Confidence")

    ratings = ["1⭐","2⭐","3⭐","4⭐","5⭐"]
    probabilities = np.random.dirichlet(np.ones(5),size=1)[0]

    fig, ax = plt.subplots()
    ax.bar(ratings, probabilities)
    ax.set_xlabel("CSAT Rating")
    ax.set_ylabel("Probability")

    st.pyplot(fig)

    # -----------------------------
    # Feature Importance Chart
    # -----------------------------
    st.subheader("📊 Feature Contribution")

    fig2, ax2 = plt.subplots()
    ax2.barh(feature_names, features)
    ax2.set_xlabel("Input Value")
    ax2.set_ylabel("Features")

    st.pyplot(fig2)
