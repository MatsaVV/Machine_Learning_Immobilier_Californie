import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import r2_score

model = joblib.load('/home/mathieu/code/Silicon_valley/model/pipe.joblib')

st.title('Prédiction de prix immobilier')

uploaded_file = st.file_uploader("Chargez vos données CSV", type="csv")
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    if 'median_house_value' in input_df.columns:
        actual_values = input_df['median_house_value']
        input_df = input_df.drop('median_house_value', axis=1)
    else:
        st.error("Le CSV doit inclure la colonne 'median_house_value' pour le calcul du R^2.")
        actual_values = None
    st.write(input_df)
else:
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

    input_data = pd.DataFrame([[Unnamed_0, longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity]],
                              columns=['Unnamed: 0', 'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity'])

if st.button('Prédire la Valeur de la Maison'):
    if uploaded_file is not None:
        prediction = model.predict(input_df)
        st.success(f'La valeur médiane prédite de la maison est: ${prediction[0]:,.2f}')
        if actual_values is not None:
            score = r2_score(actual_values, prediction)
            st.write(f"Le score R^2 de la prédiction sur les données chargées est : {score:.2f}")
    else:
        prediction = model.predict(input_data)
        st.success(f'La valeur médiane prédite de la maison est: ${prediction[0]:,.2f}')
