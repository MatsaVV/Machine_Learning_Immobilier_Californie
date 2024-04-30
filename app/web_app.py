import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import r2_score

def load_model(path):
    return joblib.load(path)

def process_csv(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'median_house_value' in df.columns:
            actual_values = df['median_house_value']
            df = df.drop('median_house_value', axis=1)
        else:
            st.error("Le CSV doit inclure la colonne 'median_house_value' pour le calcul du R^2.")
            actual_values = None
        return df, actual_values
    return None, None

def user_input_features():
    inputs = {
        'Unnamed: 0': st.number_input('Unnamed: 0'),
        'longitude': st.number_input('Longitude'),
        'latitude': st.number_input('Latitude'),
        'housing_median_age': st.number_input('Âge médian des logements'),
        'total_rooms': st.number_input('Nombre total de pièces'),
        'total_bedrooms': st.number_input('Nombre total de chambres'),
        'population': st.number_input('Population'),
        'households': st.number_input('Ménages'),
        'median_income': st.number_input('Revenu médian'),
        'ocean_proximity': st.selectbox('Proximité de l\'océan', ('NEAR BAY', 'INLAND', '<1H OCEAN', 'NEAR OCEAN', 'ISLAND'))
    }
    return pd.DataFrame([inputs])

def display_predictions(model, data, actual_values=None):
    if data is not None:
        prediction = model.predict(data)
        st.success(f'La valeur médiane prédite de la maison est: ${prediction[0]:,.2f}')
        if actual_values is not None:
            score = r2_score(actual_values, prediction)
            st.write(f"Le score R^2 de la prédiction sur les données chargées est : {score:.2f}")

def main():
    st.title('Prédiction de prix immobilier')
    model = load_model('/home/mathieu/code/Silicon_valley/model/pipe.joblib')
    uploaded_file = st.file_uploader("Chargez vos données CSV", type="csv")
    input_df, actual_values = process_csv(uploaded_file)

    if uploaded_file is not None and input_df is not None:
        st.write(input_df)
    else:
        input_df = user_input_features()

    if st.button('Prédire la Valeur de la Maison'):
        display_predictions(model, input_df, actual_values)

if __name__ == "__main__":
    main()
