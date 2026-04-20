import streamlit as st
import pandas as pd
import pickle

# Charger les objets
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("🎓 Prédiction du risque d'abandon scolaire")

# Inputs utilisateur
age = st.slider("Âge", 15, 30)
average_grade = st.slider("Moyenne", 0.0, 20.0)
absenteeism_rate = st.slider("Taux d'absence", 0.0, 0.5)
study_time_hours = st.slider("Temps d'étude", 0.0, 5.0)

gender = st.selectbox("Genre", ["Male", "Female"])
internet_access = st.selectbox("Internet", ["Yes", "No"])
extra_activities = st.selectbox("Activités", ["Yes", "No"])

# Création input
input_data = {
    "age": age,
    "average_grade": average_grade,
    "absenteeism_rate": absenteeism_rate,
    "study_time_hours": study_time_hours,
    "gender_Male": 1 if gender == "Male" else 0,
    "internet_access_Yes": 1 if internet_access == "Yes" else 0,
    "extra_activities_Yes": 1 if extra_activities == "Yes" else 0
}

input_df = pd.DataFrame([input_data])

# Ajouter colonnes manquantes
for col in columns:
    if col not in input_df:
        input_df[col] = 0

input_df = input_df[columns]

# Normalisation
input_scaled = scaler.transform(input_df)

# Prédiction
if st.button("Prédire"):
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Risque élevé ({proba*100:.1f}%)")
    else:
        st.success(f"✅ Faible risque ({(1-proba)*100:.1f}%)")