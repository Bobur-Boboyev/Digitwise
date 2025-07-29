import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle


st.set_page_config(page_title="DigitWise", layout="centered")
st.title("🧠 DigitWise: Smart Handwriting Recognizer")
st.markdown("Draw a digit (0-9) below and let the AI recognize it!")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=30,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

with open("models/MLP.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/MinMaxScaler.pkl", "rb") as f:
    scaler = pickle.load(f)

if st.button("🔍 Predict"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        img = cv2.resize(img.astype("uint8"), (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(1, -1)
        img = scaler.transform(img)
        pred = model.predict(img)

        st.subheader(f"🎯 Predicted Digit: {pred[0]}")
