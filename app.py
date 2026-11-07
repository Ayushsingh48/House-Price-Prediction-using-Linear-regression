import streamlit as st, pandas as pd, joblib

model = joblib.load("house_price_lr_model.pkl")
st.title("House Price Prediction (Linear Regression)")

c1, c2 = st.columns(2)
longitude = c1.number_input("Longitude", -124.35, -114.31, -119.57)
latitude = c2.number_input("Latitude", 32.54, 41.95, 35.63)
age = c1.number_input("Housing median age", 1, 52, 29)
rooms = c2.number_input("Total rooms", 1, 40000, 2127)
bedrooms = c1.number_input("Total bedrooms", 1, 7000, 435)
population = c2.number_input("Population", 1, 36000, 1166)
households = c1.number_input("Households", 1, 6100, 409)
income = c2.number_input("Median income (x $10,000)", 0.5, 15.0, 3.5)
ocean = st.selectbox("Ocean proximity", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])

if st.button("Predict Price"):
    row = pd.DataFrame([{"longitude": longitude, "latitude": latitude,
        "housing_median_age": age, "total_rooms": rooms, "total_bedrooms": bedrooms,
        "population": population, "households": households,
        "median_income": income, "ocean_proximity": ocean}])
    st.success(f"Predicted median house value: ${model.predict(row)[0]:,.0f}")
