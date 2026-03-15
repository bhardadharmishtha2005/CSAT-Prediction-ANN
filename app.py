import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import numpy as np

@st.cache_resource
def load_my_model():
    try:
        # 1. Manually Rebuild the Architecture based on your Colab logs
        model = Sequential([
            Input(shape=(12,)),  # Matches your [None, 12] input_shape
            Dense(96, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(5, activation='softmax')
        ])
        
        # 2. Load only the weights (this avoids the 'quantization_config' error)
        model.load_weights('csat_model.keras')
        return model
    except Exception as e:
        # Fallback: If weight loading fails, try standard load
        try:
            return tf.keras.models.load_model('csat_model.keras', compile=False)
        except:
            st.error(f"Critical Load Error: {e}")
            return None
