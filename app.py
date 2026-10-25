import os
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Rainfall Prediction", page_icon="🌧️", layout="centered")

MODEL_FILE = os.path.join(os.path.dirname(__file__), "weather_rainfall_model.pkl")
TYPICAL_RMSE_MM = 8.1   # cross-validation RMSE of the best model (from the notebook)
DIRECTIONS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


st.title("🌧️ Tomorrow's Rainfall Predictor")
st.write("Enter **today's** weather measurements. The model predicts how many **millimetres of rain** to expect **tomorrow**.")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'weather_rainfall_model.pkl' not found. Put it in the same folder as app.py.")
    st.stop()

st.subheader("Temperature (°C)")
c1, c2 = st.columns(2)
min_temp = c1.number_input("Minimum temperature", 0.0, 40.0, 15.0, 0.1)
max_temp = c2.number_input("Maximum temperature", 5.0, 50.0, 23.0, 0.1)
temp9 = c1.number_input("Temperature at 9am", 0.0, 45.0, 18.0, 0.1)
temp3 = c2.number_input("Temperature at 3pm", 5.0, 50.0, 21.0, 0.1)

st.subheader("Rain, sun and evaporation")
c1, c2, c3 = st.columns(3)
rainfall = c1.number_input("Today's rainfall (mm)", 0.0, 200.0, 0.0, 0.1)
evaporation = c2.number_input("Evaporation (mm)", 0.0, 25.0, 5.0, 0.1)
sunshine = c3.number_input("Sunshine (hours)", 0.0, 14.0, 7.0, 0.1)

st.subheader("Wind")
c1, c2, c3 = st.columns(3)
gust_dir = c1.selectbox("Wind gust direction", DIRECTIONS, index=DIRECTIONS.index("W"))
dir9 = c2.selectbox("Wind direction 9am", DIRECTIONS, index=DIRECTIONS.index("W"))
dir3 = c3.selectbox("Wind direction 3pm", DIRECTIONS, index=DIRECTIONS.index("E"))
gust_speed = c1.number_input("Wind gust speed (km/h)", 0, 120, 41)
speed9 = c2.number_input("Wind speed 9am (km/h)", 0, 80, 15)
speed3 = c3.number_input("Wind speed 3pm (km/h)", 0, 80, 19)

st.subheader("Humidity, pressure and cloud")
c1, c2 = st.columns(2)
hum9 = c1.slider("Humidity 9am (%)", 0, 100, 68)
hum3 = c2.slider("Humidity 3pm (%)", 0, 100, 55)
pres9 = c1.number_input("Pressure 9am (hPa)", 960.0, 1060.0, 1018.0, 0.1)
pres3 = c2.number_input("Pressure 3pm (hPa)", 960.0, 1060.0, 1016.0, 0.1)
cloud9 = c1.slider("Cloud cover 9am (oktas, 0-8)", 0, 8, 4)
cloud3 = c2.slider("Cloud cover 3pm (oktas, 0-8)", 0, 8, 4)

month_name = st.selectbox("Month", MONTHS, index=6)

if st.button("Predict rainfall", type="primary"):
    row = {
        "MinTemp": min_temp, "MaxTemp": max_temp, "Temp9am": temp9, "Temp3pm": temp3,
        "Rainfall": rainfall, "Evaporation": evaporation, "Sunshine": sunshine,
        "WindGustSpeed": gust_speed, "WindSpeed9am": speed9, "WindSpeed3pm": speed3,
        "Humidity9am": hum9, "Humidity3pm": hum3,
        "Pressure9am": pres9, "Pressure3pm": pres3,
        "Cloud9am": cloud9, "Cloud3pm": cloud3, "Month": MONTHS.index(month_name) + 1,
        "WindGustDir": gust_dir, "WindDir9am": dir9, "WindDir3pm": dir3,
    }
    # same column order as in training (the pipeline selects columns by name)
    features = pd.DataFrame([row])
    prediction = max(0.0, float(model.predict(features)[0]))

    st.success(f"Predicted rainfall tomorrow: **{prediction:.1f} mm**")
    if prediction < 1:
        st.info("☀️ Little or no rain expected.")
    elif prediction < 10:
        st.info("🌦️ Light to moderate rain expected.")
    elif prediction < 30:
        st.warning("🌧️ Heavy rain expected.")
    else:
        st.error("⛈️ Very heavy rain expected.")
    st.caption(f"Typical error of this model is about ±{TYPICAL_RMSE_MM} mm "
               "(rainfall is hard to predict; treat this as an estimate).")
