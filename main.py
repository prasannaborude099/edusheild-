import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load model and scaler
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or scaler file not found! Please upload model.pkl and scaler.pkl.")
    st.stop()

st.title("EduShield AI: Dropout Predictor & Counsellor")
st.write("Enter student details to predict dropout risk and get advice")

# Input form
marital = st.selectbox("Marital Status", [1, 2, 3, 4, 5, 6], help="1=Single, 2=Married, etc.")
app_mode = st.slider("Application Mode", 1, 44, 16, help="Admission type (1-44)")
course = st.slider("Course", 1, 17, 2, help="Course ID (1-17)")
attendance = st.selectbox("Daytime/Evening Attendance", [0, 1], help="0=Evening, 1=Daytime")
prev_qual = st.slider("Previous Qualification", 1, 19, 1, help="Previous degree (1-19)")
nationality = st.slider("Nacionality", 1, 22, 1, help="Nationality code (1-22)")
mother_qual = st.slider("Mother's Qualification", 1, 44, 1, help="Mother's education (1-44)")
father_qual = st.slider("Father's Qualification", 1, 44, 1, help="Father's education (1-44)")
mother_occ = st.slider("Mother's Occupation", 0, 9, 1, help="Mother's job (0-9)")
father_occ = st.slider("Father's Occupation", 0, 9, 1, help="Father's job (0-9)")
grade = st.number_input("1st Semester Grade", 0.0, 20.0, 12.0, help="Grade in 1st semester (0-20)")
displaced = st.selectbox("Displaced", [0, 1], help="0=Not displaced, 1=Displaced")
special_needs = st.selectbox("Educational Special Needs", [0, 1], help="0=No, 1=Yes")
debtor = st.selectbox("Debtor", [0, 1], help="0=No debt, 1=Has debt")
fees = st.selectbox("Tuition Fees Up to Date", [0, 1], help="0=Not paid, 1=Paid")
gender = st.selectbox("Gender", [0, 1], help="0=Male, 1=Female")
scholarship = st.selectbox("Scholarship Holder", [0, 1], help="0=No, 1=Yes")
age = st.slider("Age at Enrollment", 17, 70, 18, help="Age when enrolled")
international = st.selectbox("International", [0, 1], help="0=Local, 1=International")

student_data = {
    'Marital status': marital, 'Application mode': app_mode, 'Course': course,
    'Daytime/evening attendance': attendance, 'Previous qualification': prev_qual,
    'Nacionality': nationality, "Mother's qualification": mother_qual,
    "Father's qualification": father_qual, "Mother's occupation": mother_occ,
    "Father's occupation": father_occ, 'Curricular units 1st sem (grade)': grade,
    'Displaced': displaced, 'Educational special needs': special_needs, 'Debtor': debtor,
    'Tuition fees up to date': fees, 'Gender': gender, 'Scholarship holder': scholarship,
    'Age at enrollment': age, 'International': international
}

features = [
    'Marital status', 'Application mode', 'Course', 'Daytime/evening attendance',
    'Previous qualification', 'Nacionality', "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", 'Curricular units 1st sem (grade)',
    'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date',
    'Gender', 'Scholarship holder', 'Age at enrollment', 'International'
]

# Predict
input_data = np.array([student_data[f] for f in features]).reshape(1, -1)
risk = model.predict_proba(scaler.transform(input_data))[0][1] * 100

st.write(f"**Dropout Risk: {risk:.1f}%**")
if risk > 50:
    st.write("**Counselling: High risk!**")
    st.write("- Apply for scholarship or fee waiver immediately.")
    st.write("- Clear pending fees to reduce stress.")
    st.write("- Join study groups or tutoring for academic support.")
    st.write(f"**Story**: If you tackle fees and attendance, you'll graduate like a champ!")
else:
    st.write("**Counselling: Low risk!** Keep it up with weekly check-ins.")
