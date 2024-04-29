import streamlit as st
import pandas as pd
import joblib

model = joblib.load('/home/mathieu/code/Silicon_valley/model/pipe.joblib')

st.title('Prédiction de prix immobilier')
Unnamed_0 = st.number_input('Unnamed: 0')
longitude = st.number_input('Longitude')
latitude = st.number_input('Latitude')
housing_median_age = st.number_input('Âge médian des logements')
total_rooms = st.number_input('Nombre total de pièces')
total_bedrooms = st.number_input('Nombre total de chambres')
population = st.number_input('Population')
households = st.number_input('Ménages')
median_income = st.number_input('Revenu médian')
ocean_proximity = st.selectbox('Proximité de l\'océan', ('NEAR BAY', 'INLAND', '<1H OCEAN', 'NEAR OCEAN', 'ISLAND'))

if st.button('Prédire la Valeur de la Maison'):
    input_data = pd.DataFrame([[Unnamed_0, longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity]],
                                  columns=['Unnamed: 0', 'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity'])

    prediction = model.predict(input_data)
    st.success(f'La valeur médiane prédite de la maison est: ${prediction[0]}')
